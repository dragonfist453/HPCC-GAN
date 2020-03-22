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
#option('outputLimit',200);

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
//Must check for errors and correct them as mucn as possible. Please do work, I pray to gods for this ;_;

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
                        '''layers.Reshape((28,28,1))'''];
                    
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
        fldef_combined := DATASET([{'noise','''layers.Input(shape=(100,))''',[]},              //Input of Generator
                                {'g1','''layers.Dense(256, input_dim=100)''',['noise']},        //Generator layer 1
                                {'g2','''layers.LeakyReLU(alpha=0.2)''',['g1']},                //Generator layer 2
                                {'g3','''layers.BatchNormalization(momentum=0.8)''',['g2']},    //Generator layer 3
                                {'g4','''layers.Dense(512)''',['g3']},                          //Generator layer 4
                                {'g5','''layers.LeakyReLU(alpha=0.2)''',['g4']},                //Generator layer 5
                                {'g6','''layers.BatchNormalization(momentum=0.8)''',['g5']},    //Generator layer 6
                                {'g7','''layers.Dense(1024)''',['g6']},                         //Generator layer 7
                                {'g8','''layers.LeakyReLU(alpha=0.2)''',['g7']},                //Generator layer 8
                                {'g9','''layers.BatchNormalization(momentum=0.8)''',['g8']},    //Generator layer 9
                                {'g10','''layers.Dense(784,activation='tanh')''',['g9']},       //Generator layer 10
                                {'img','''layers.Reshape((28,28,1))''',['g10']},                //Generate output
                                {'d1','''layers.Flatten(input_shape=(28,28,1),trainable = False)''',['img']}, //Discriminator layer 1
                                {'d2','''layers.Dense(512,trainable = False)''',['d1']},   //Discriminator layer 2
                                {'d3','''layers.LeakyReLU(alpha=0.2,trainable = False)''',['d2']},                //Discriminator layer 3
                                {'d4','''layers.Dense(256,trainable = False)''',['d3']},                          //Discriminator layer 4
                                {'d5','''layers.LeakyReLU(alpha=0.2,trainable = False)''',['d4']},                //Discriminator layer 5
                                {'validity','''layers.Dense(1,activation='sigmoid',trainable = False)''',['d5']}],//Output of Discriminator, valid image or not
                        FuncLayerDef);

        compiledef_combined := '''compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';

        //Define combined network
        combined := GNNI.DefineFuncModel(session, fldef_combined, ['noise'],['validity'],compiledef_combined); //Combined model definition
        com_def := OUTPUT(combined, NAMED('combined_id'));
        

        //Dataset of 1s for classification
        valid_data := DATASET(batchSize, TRANSFORM(TensData,
                        SELF.indexes := [COUNTER, 1],
                        SELF.value := 1));
        valid := Tensor.R4.MakeTensor([0,batchSize],valid_data);

        //Dataset of 0s for classification
        fake_data := DATASET(batchSize, TRANSFORM(TensData,
                        SELF.indexes := [COUNTER, 1],
                        SELF.value := 0));
        fake := Tensor.R4.MakeTensor([0,batchSize],fake_data);   

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
                loopDiscriminator := discriminator + epochNum;
                loopCombined := combined + epochNum;
                loopGenerator := generator + epochNum;

                //Split weights accordingly. Generator layer <=20. Discriminator layers > 20. Discriminator must be subtracted by 20 to get its proper weights
                genWts := SORT(wts(wi <= (Tensor.t_WorkItem) 20), wi, sliceid, LOCAL);
                splitdisWts := SORT(wts(wi > (Tensor.t_WorkItem) 20), wi, sliceid, LOCAL);
                diswts := PROJECT(splitdisWts, TRANSFORM(t_Tensor,
                                        SELF.wi := LEFT.wi - 20,
                                        SELF := LEFT
                                        ));

                //Setting new weights
                generator1 := GNNI.SetWeights(loopGenerator + 1, genWts);
                discriminator1 := GNNI.SetWeights(loopDiscriminator + 1, disWts); 

                //Fitting real data
                discriminator2 := GNNI.Fit(discriminator1 + 1, X_dat, valid, 1, batchSize);

                //Predicting using Generator for fake images
                gen_X_dat1 := GNNI.Predict(generator1, train_noise1);
                
                //Fitting random data
                discriminator3 := GNNI.Fit(discriminator2 + 1, gen_X_dat1, fake, 1, batchSize);

                //Noise for generator to make fakes
                random_data2 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                train_noise2 := Tensor.R4.MakeTensor([0,latentDim], random_data2);

                //Get discriminator weights, add 20 to it, change discriminator weights of combined model, set combined weights
                updateddisWts := GNNI.GetWeights(discriminator3);
                newdisWts := PROJECT(updateddisWts, TRANSFORM(t_Tensor,
                                        SELF.wi := LEFT.wi + 20,
                                        SELF := LEFT
                                        ));
                comWts := SORT(newdisWts(wi > (Tensor.t_WorkItem) 20) + wts(wi <= (Tensor.t_WorkItem) 20), wi, sliceid, LOCAL);
                combined1 := GNNI.SetWeights(loopCombined, comWts);

                //Train generator using combined model
                combined2 := GNNI.Fit(combined1, train_noise2, valid, 1, batchSize);

                //Get combined weights to return
                newWts := GNNI.GetWeights(combined2);
                
                //Logging progress when done
                logProgress := Syslog.addWorkunitInformation('GAN training - Epoch : '+epochNum);

                //List of actions to do in order before log progress and returning weights
                actions := ORDERED(discriminator1, discriminator2, generator1, discriminator3, combined1, combined2, logProgress);

                RETURN WHEN(newWts, actions, BEFORE);
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

generator := GAN_train(trainX);

generated := GNNI.Predict(generator, train_noise);

generated_data := Tensor.R4.GetData(generated);

OUTPUT(generated_data);