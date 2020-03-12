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
//Format of the image
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
batchSize := 128;

//Take MNIST dataset using IMG module
mnist_train_images := IMG.MNIST_train_image();

//Tensor dataset having image data normalised to range of -1 to 1
trainX0 := NORMALIZE(mnist_train_images, imgSize, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28+1, (COUNTER-1)%28+1, 1],
                            SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 )); 

//Random set of normal data
random_data := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV batchSize + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                     
//Builds tensors for the neural network
trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], trainX0); 
train_noise := Tensor.R4.MakeTensor([batchSize,latentDim], random_data);


//Function to be modified as the logic of GANs is incorrect below. Generator does not train which is bad
//Other than that, this is ideal function which must be optimised to add to GNN module. It takes what is required and rest is to be taken care of inside the function.
//Optimise further after basic completion and discussion with Roger. This will go awesome. Believe in yourself :")

//Please address later about how users can change the layers efficiently. VERY IMPORTANT.
//Returns model ID to predict using the GAN
UNSIGNED4 GAN_train(DATASET(t_Tensor) input,
                        UNSIGNED4 batchSize = 100,
                        UNSIGNED4 numEpochs = 1) := FUNCTION

        //Secret item for later ;)
        recordCount := COUNT(input);

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
                                {'img','''layers.Reshape((1,28,28,1))''',['g10']},                //Generate output
                                {'d1','''layers.Flatten(input_shape=(28,28,1))''',['img']}, //Discriminator layer 1
                                {'d2','''layers.Dense(512)''',['d1']},   //Discriminator layer 2
                                {'d3','''layers.LeakyReLU(alpha=0.2)''',['d2']},                //Discriminator layer 3
                                {'d4','''layers.Dense(256)''',['d3']},                          //Discriminator layer 4
                                {'d5','''layers.LeakyReLU(alpha=0.2)''',['d4']},                //Discriminator layer 5
                                {'validity','''layers.Dense(1,activation='sigmoid')''',['d5']}],//Output of Discriminator, valid image or not
                        FuncLayerDef);

        compiledef_combined := '''compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';

        //Define combined network
        combined := GNNI.DefineFuncModel(session, fldef_combined, ['noise'],['validity'],compiledef_combined); //Combined model definition


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

        //The required weights
        combWts := GNNI.GetWeights(combined);
        discWts := GNNI.GetWeights(discriminator);

        //Combine wts together to form one set
        max_comb_wi := MAX(combWts, wi);
        adjDiscWts := PROJECT(discWts, TRANSFORM(RECORDOF(LEFT),
                                        SELF.wi := LEFT.wi + max_comb_wi,
                                        SELF := LEFT
                                        ));
        wts := combWts + adjDiscWts;                                

        DATASET(t_Tensor) train(DATASET(t_Tensor) wts, UNSIGNED4 epochNum) := FUNCTION
                //Random position in Tensor which is (batchSize) less than COUNT(input)
                batchPos := RANDOM()%recordCount - batchSize;

                //Extract (batchSize) tensors starting from a random batchPos from the tensor input. Now we have a random input images of (batchSize) rows.
                X_dat := int.TensExtract(input, batchPos, batchSize);

                //Noise for generator to make fakes
                random_data1 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV batchSize + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                train_noise1 := Tensor.R4.MakeTensor([batchSize,latentDim], random_data1);

                //Extract individual weights from the combined tensor
                comWts := wts(wi<=max_comb_wi);
                disWts := PROJECT(wts(wi>max_comb_wi), TRANSFORM(RECORDOF(LEFT),
                                                        SELF.wi := LEFT.wi-max_comb_wi,
                                                        SELF := LEFT
                                                        ));

                //New model IDs
                loopDiscriminator := discriminator + epochNum;
                loopCombined := combined + epochNum;

                //Setting new weights
                discriminator1 := GNNI.SetWeights(loopDiscriminator, disWts); 
                combined1 := GNNI.SetWeights(loopCombined, comWts);

                //Fitting real data
                discriminator2 := GNNI.Fit(discriminator1, X_dat, valid, 1, batchSize);

                //Predicting using Generator for fake images
                gen_X_dat1 := GNNI.Predict(generator, train_noise1);
                
                //Fitting random data
                discriminator3 := GNNI.Fit(discriminator2, gen_X_dat1, fake, 1, batchSize);

                //Noise for generator to make fakes
                random_data2 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV batchSize + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                train_noise2 := Tensor.R4.MakeTensor([batchSize,latentDim], random_data2);

                //Train generator using combined model
                combined2 := GNNI.Fit(combined1, train_noise, valid, 1, batchSize);

                //Get weights of the models
                newcombWts := GNNI.GetWeights(combined2);
                newdiscWts := GNNI.GetWeights(discriminator3);

                //Combine wts together to form one set
                max_com_wi := MAX(newcombWts, wi);
                adjDisWts := PROJECT(newdiscWts, TRANSFORM(RECORDOF(LEFT),
                                                SELF.wi := LEFT.wi + max_com_wi,
                                                SELF := LEFT
                                                ));
                newWts := newcombWts + adjDisWts;

                //gen_loss := IF(EXISTS(newWts), GNNI.GetLoss(generator), 0);
                //dis_loss := IF(EXISTS(newWts), GNNI.GetLoss(discriminator_fooled), 0);
                
                logProgress := Syslog.addWorkunitInformation('GAN training - Epoch : '+epochNum
                //+' Generator loss : '+gen_loss+' Discriminator loss : '+dis_loss
                );
                RETURN WHEN(newWts, logProgress);
        END;        

        finalWts := LOOP(wts, ROUNDUP(numEpochs), train(ROWS(LEFT),COUNTER));

        //Extract individual weights from the combined tensor
        comWts := wts(wi<=max_comb_wi);
        disWts := PROJECT(wts(wi>max_comb_wi), TRANSFORM(RECORDOF(LEFT),
                                                SELF.wi := LEFT.wi-max_comb_wi,
                                                SELF := LEFT
                                                ));

        //Final model IDs
        finalDiscriminator := discriminator + numEpochs + 1;
        finalCombined := combined + numEpochs + 1;

        //Setting new weights
        discriminator_trained := GNNI.SetWeights(finalDiscriminator, disWts); 
        combined_trained := GNNI.SetWeights(finalCombined, comWts);

        RETURN combined_trained;
END;        

final := GAN_train(trainX);

generated := GNNI.Predict(final+2, train_noise);

generated_data := Tensor.R4.GetData(generated);

OUTPUT(generated_data);