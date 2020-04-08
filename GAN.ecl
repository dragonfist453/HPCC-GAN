IMPORT Python3 AS Python;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Thorlib;
IMPORT Std.System.Log AS Syslog;
IMPORT IMG.IMG;
IMPORT GNN.Utils;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;
FuncLayerDef := Types.FuncLayerDef;

RAND_MAX := POWER(2,32) - 1;
#option('outputLimit',2000);

//Input and Preprocessing
IMG_FORMAT := RECORD
    UNSIGNED id;
    DATA image;
END;

//Train data definitions
imgcount_train := 60000;
imgRows := 28;
imgCols := 28;
imgChannels := 1;
imgSize := imgRows * imgCols;
latentDim := 100;
numClasses := 10;
batchSize := 100;

//Take MNIST dataset using IMG module
mnist_train_images := IMG.MNIST_train_image();

//Tensor dataset having image data normalised to range of -1 to 1
trainX0 := NORMALIZE(mnist_train_images, imgSize, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28+1, (COUNTER-1)%28+1, 1],
                            SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 )); 

//Random set of normal data
random_data := DATASET(latentDim, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                     
//Builds tensors for the neural network
trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], trainX0); 
train_noise := Tensor.R4.MakeTensor([0,latentDim], random_data);


//Function's logic seems to be perfect after integration of transfer of weights and non-trainable layers. 

//Please address later about how users can change the layers efficiently. VERY IMPORTANT.
//Returns model ID to predict using the GAN
UNSIGNED4 GAN_train(DATASET(t_Tensor) input,
                        UNSIGNED4 batchSize = 100,
                        UNSIGNED4 numEpochs = 1) := FUNCTION

        //Secret item for later ;)
        recordCount := TENSOR.R4.GetRecordCount(input);

        //Start session for GAN
        session := GNNI.GetSession();

        
        //GENERATOR
        //Generator model definition information
        ldef_generator := ['''layers.Input(shape=(100,))''',
                        '''layers.Dense(256, input_dim=100)''',
                        '''layers.LeakyReLU(alpha=0.2)''',    
                        '''layers.BatchNormalization(momentum=0.8)''',
                        '''layers.Dense(512)''',
                        '''layers.LeakyReLU(alpha=0.2)''',
                        '''layers.BatchNormalization(momentum=0.8)''',
                        '''layers.Dense(1024)''',
                        '''layers.LeakyReLU(alpha=0.2)''',
                        '''layers.BatchNormalization(momentum=0.8)''',
                        '''layers.Dense(784,activation='tanh')''',
                        '''layers.Reshape((1,28,28,1))'''];
                    
        compiledef_generator := '''compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';

        //Define generator network
        generator := GNNI.DefineModel(session, ldef_generator, compiledef_generator); //Generator model definition
        gen_def := OUTPUT(generator, NAMED('generater_id'));


        //DISCRIMINATOR
        //Discriminator model definition information
        ldef_discriminator := ['''layers.Input(shape=(28,28,1))''',
                                '''layers.Flatten(input_shape=(28,28,1))''',
                                '''layers.Dense(512)''',
                                '''layers.LeakyReLU(alpha=0.2)''',
                                '''layers.Dense(256)''',
                                '''layers.LeakyReLU(alpha=0.2)''',
                                '''layers.Dense(1,activation='sigmoid')'''];

        compiledef_discriminator := '''compile(loss='binary_crossentropy',
                                        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                                        metrics=['accuracy'])''';                       
   
        //Define discriminator network
        discriminator := GNNI.DefineModel(session, ldef_discriminator, compiledef_discriminator); //Discriminator model definition
        dis_def := OUTPUT(discriminator, NAMED('discriminator_id'));


        //COMBINED functional model
        //Combined model definition information
        ldef_combined := ['''layers.Input(shape=(100,))''', 
                        '''layers.Dense(256, input_dim=100)''',//1
                        '''layers.LeakyReLU(alpha=0.2)''',    //3
                        '''layers.BatchNormalization(momentum=0.8)''',//6
                        '''layers.Dense(512)''',    //7
                        '''layers.LeakyReLU(alpha=0.2)''',  //9
                        '''layers.BatchNormalization(momentum=0.8)''',  //12
                        '''layers.Dense(1024)''',   //13
                        '''layers.LeakyReLU(alpha=0.2)''',  //15
                        '''layers.BatchNormalization(momentum=0.8)''',  //18
                        '''layers.Dense(784,activation='tanh')''',  //19
                        '''layers.Reshape((1,28,28,1))''', //20
                        '''layers.Flatten(input_shape=(28,28,1), trainable=False)''',//1
                        '''layers.Dense(512,trainable=False)''',//2
                        '''layers.LeakyReLU(alpha=0.2, trainable=False)''',//3
                        '''layers.Dense(256,trainable=False)''',//4
                        '''layers.LeakyReLU(alpha=0.2, trainable=False)''',//5
                        '''layers.Dense(1,activation='sigmoid',trainable=False)'''];//6

        compiledef_combined := '''compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';
        
        //Define combined functional network
        combined := GNNI.DefineModel(session, ldef_combined, compiledef_combined);
        combined_def := OUTPUT(combined, NAMED('combined_id'));

        //Dataset of 1s for classification
        valid_data := DATASET(batchSize, TRANSFORM(TensData,
                        SELF.indexes := [COUNTER, 1],
                        SELF.value := 1));
        valid := Tensor.R4.MakeTensor([0,1],valid_data);

        //Dataset of 0s for classification
        fake_data := DATASET(batchSize, TRANSFORM(TensData,
                        SELF.indexes := [COUNTER, 1],
                        SELF.value := 0));
        fake := Tensor.R4.MakeTensor([0,1],fake_data);

        //Mixed dataset of above two
        mixed_data := DATASET(batchSize*2, TRANSFORM(TensData,
                        SELF.indexes := [COUNTER,1],
                        SELF.value := IF(COUNTER <= batchSize,1,0);
        ));
        mixed := Tensor.R4.MakeTensor([0,1], mixed_data);
        Y_train := mixed;

        //Get only initial combined weights
        wts := GNNI.GetWeights(combined);             

        DATASET(t_Tensor) train(DATASET(t_Tensor) wts, UNSIGNED4 epochNum) := FUNCTION
                //Random position in Tensor which is (batchSize) less than COUNT(input)
                batchPos := RANDOM()%(recordCount - batchSize);
                
                //Extract (batchSize) tensors starting from a random batchPos from the tensor input. Now we have a random input images of (batchSize) rows.
                X_dat := int.TensExtract(input, batchPos, 100);

                //Noise for generator to make fakes
                random_data1 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                train_noise1 := Tensor.R4.MakeTensor([0,latentDim], random_data1);

                //New model IDs
                loopDiscriminator := discriminator + 2*(epochNum - 1);
                loopCombined := combined + 2*(epochNum - 1);
                loopGenerator := generator + (epochNum - 1);

                //Split weights accordingly. Generator layer <=20. Discriminator layers > 20. Discriminator must be subtracted by 20 to get its proper weights
                genWts := SORT(wts(wi <= (Tensor.t_WorkItem) 20), wi, sliceid, LOCAL);
                splitdisWts := SORT(wts(wi > (Tensor.t_WorkItem) 20), wi, sliceid, LOCAL);
                diswts := PROJECT(splitdisWts, TRANSFORM(t_Tensor,
                                        SELF.wi := LEFT.wi - 20,
                                        SELF := LEFT
                                        ));

                //Setting generator weights
                generator1 := GNNI.SetWeights(loopGenerator, genWts);

                //Predicting using Generator for fake images
                gen_X_dat1 := GNNI.Predict(generator1, train_noise1);

                gen_imgs := PROJECT(gen_X_dat1, TRANSFORM(t_Tensor,
                            SELF.shape := [0,LEFT.shape[2],LEFT.shape[3],LEFT.shape[4]],
                            SELF.wi := 1,
                            SELF := LEFT
                            ));       
                        
                gen_out := Tensor.R4.GetData(gen_imgs);

                imagerows := MAX(gen_out, indexes[2]); 
                imagecols := MAX(gen_out, indexes[3]);
                imagechannels := MAX(gen_out, indexes[4]);

                dim := imagerows*imagecols*imagechannels;

                toTensor := PROJECT(gen_out, TRANSFORM(TensData,
                                        SELF.indexes := [COUNTER DIV (dim+1) + 1 + batchSize,LEFT.indexes[2],LEFT.indexes[3],LEFT.indexes[4]],
                                        SELF := LEFT
                                        ));

                X_imgs := Tensor.R4.GetData(X_dat);

                toNN := X_imgs + toTensor;                        

                X_train := Tensor.R4.MakeTensor([0,imagerows,imagecols,imagechannels],toNN);                      

                //Setting discriminator weights
                discriminator1 := GNNI.SetWeights(loopDiscriminator, disWts); 

                //Fitting real and random data
                discriminator2 := GNNI.Fit(discriminator1, X_train, Y_train, batchSize*2, 1);

                //Noise for generator to make fakes
                random_data2 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                train_noise2 := Tensor.R4.MakeTensor([0,latentDim], random_data2);

                //Get discriminator weights, add 20 to it, change discriminator weights of combined model, set combined weights
                updateddisWts := GNNI.GetWeights(discriminator2);
                newdisWts := PROJECT(updateddisWts, TRANSFORM(t_Tensor,
                                        SELF.wi := LEFT.wi + 20,
                                        SELF := LEFT
                                        ));
                comWts := SORT(newdisWts(wi > (Tensor.t_WorkItem) 20) + wts(wi <= (Tensor.t_WorkItem) 20), wi, sliceid, LOCAL);
                combined1 := GNNI.SetWeights(loopCombined, comWts);

                //Fit combined model
                combined2 := GNNI.Fit(combined1, train_noise2, valid, batchSize, 1);

                //Get combined weights to return
                newWts := GNNI.GetWeights(combined2);
                
                //Logging progress when done
                logProgress := Syslog.addWorkunitInformation('GAN training - Epoch : '+epochNum);

                //List of actions to do in order before log progress and returning weights
                //actions := ORDERED(discriminator1, discriminator2, generator1, discriminator3, combined1, combined2, logProgress);

                RETURN WHEN(newWts, logProgress);
        END;        

        //Call loop to train numEpochs times
        finalWts := LOOP(wts, ROUNDUP(numEpochs), train(ROWS(LEFT),COUNTER));

        //Final model IDs
        finalGenerator := generator + numEpochs + 1;

        //Setting new weights
        genWts := SORT(finalWts(wi<=20), wi, sliceid, LOCAL);
        generator_trained := GNNI.SetWeights(finalGenerator, genWts);

        //Return the generator id to use generator to predict
        RETURN generator_trained;
END;        

//Get generator after training
generator := GAN_train(trainX,batchSize,100);

//Predict an image from noise
generated := GNNI.Predict(generator, train_noise);
generated_data := Tensor.R4.GetData(generated);
OUTPUT(generated_data, ,'~GAN::output_tensdata', OVERWRITE);

//Convert from tensor data to images
outputImage := IMG.TenstoImg(generated_data);

//Convert image data to jpg format to despray
mnistjpg := IMG.OutputMNIST(outputImage);

OUTPUT(mnistjpg, ,'~GAN::output_image', OVERWRITE);