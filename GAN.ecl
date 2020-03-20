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


//Function to be modified as the logic of GANs is incorrect below. Generator does not train which is bad
//Other than that, this is ideal function which must be optimised to add to GNN module. It takes what is required and rest is to be taken care of inside the function.
//Optimise further after basic completion and discussion with Roger. This will go awesome. Believe in yourself :")

//Please address later about how users can change the layers efficiently. VERY IMPORTANT.
//Returns model ID to predict using the GAN
UNSIGNED4 GAN_train(DATASET(t_Tensor) input,
                        UNSIGNED4 batchSize = 100,
                        UNSIGNED4 numEpochs = 1) := FUNCTION

        //Secret item for later ;)
        recordCount := TENSOR.R4.GetRecordCount(input);

        //Start session for GAN
        session := GNNI.GetSession();

        //Functional model
        //Functional model definition information
        fldef := DATASET([{'noise','''layers.Input(shape=(100,))''',[]},              //Input of Generator
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
                        {'img_out','''layers.Reshape((1,28,28,1))''',['g10']},                //Generate output
                        {'img_in','''layers.Flatten(input_shape=(28,28,1))''',['img_out']}, //Input of Discriminator
                        {'d1','''layers.Dense(512)''',['img_in']},   //Discriminator layer 1
                        {'d2','''layers.LeakyReLU(alpha=0.2)''',['d1']},                //Discriminator layer 2
                        {'d3','''layers.Dense(256)''',['d2']},                          //Discriminator layer 3
                        {'d4','''layers.LeakyReLU(alpha=0.2)''',['d3']},                //Discriminator layer 4
                        {'validity','''layers.Dense(1,activation='sigmoid')''',['d4']}],//Output of Discriminator, valid image or not
                        FuncLayerDef);

        compiledef := '''compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';

        //Define generator functional network
        generator := GNNI.DefineFuncModel(session, fldef, ['noise'],['img_out'],compiledef); //Generator model definition
        generator_def := OUTPUT(generator, NAMED('generator_id'));

        //Define discriminator functional network
        discriminator := GNNI.DefineFuncModel(session, fldef, ['img_in'], ['validity'], compiledef); //Discriminator model definition
        discriminator_def := OUTPUT(discriminator, NAMED('discriminator_id'));
        
        //Define combined functional network
        combined := GNNI.DefineFuncModel(session, fldef, ['noise'], ['validity'], compiledef);
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

        //The required weights
        wts := GNNI.GetWeights(discriminator);

        DATASET(t_Tensor) train(DATASET(t_Tensor) wts, UNSIGNED4 epochNum) := FUNCTION
                //Random position in Tensor which is (batchSize) less than COUNT(input)
                batchPos := RANDOM()%(recordCount - batchSize);
                
                //Extract (batchSize) tensors starting from a random batchPos from the tensor input. Now we have a random input images of (batchSize) rows.
                X_dat := int.TensExtract(input, batchPos, 100);

                //Noise for generator to make fakes
                random_data1 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV batchSize + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                train_noise1 := Tensor.R4.MakeTensor([0,latentDim], random_data1);

                //New model IDs
                loopDiscriminator := discriminator + epochNum;
                loopCombined := combined + epochNum;
                loopGenerator := combined + epochNum;

                //Setting new weights
                discriminator1 := GNNI.SetWeights(loopDiscriminator, wts); 
                generator1 := GNNI.SetWeights(loopGenerator, wts);

                //Fitting real data
                discriminator2 := GNNI.Fit(discriminator1, X_dat, valid, batchSize, 1); //Some problem with weight dimensions. Check weight dimensions. How to output in ECL loop?

                //Predicting using Generator for fake images
                gen_X_dat := GNNI.Predict(generator1, train_noise1);

                //Project the correct shape onto tensor
                gen_X_dat1 := PROJECT(gen_X_dat, TRANSFORM(t_Tensor,
                        SELF.nodeId := LEFT.nodeId,
                        SELF.wi := LEFT.wi,
                        SELF.sliceid := LEFT.sliceid,
                        SELF.shape := [0,LEFT.shape[2],LEFT.shape[3],LEFT.shape[4]],
                        SELF.dataType := LEFT.dataType,
                        SELF.maxSliceSize := LEFT.maxSliceSize,
                        SELF.sliceSize := LEFT.sliceSize,
                        SELF.denseData := LEFT.denseData,
                        SELF.sparseData := LEFT.sparseData
                        ));
                
                //Fitting random data
                discriminator3 := GNNI.Fit(discriminator2, gen_X_dat1, fake, batchSize, 1);

                //Noise for generator to make fakes
                random_data2 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV batchSize + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                train_noise2 := Tensor.R4.MakeTensor([0,latentDim], random_data2);

                //Take weights to set combined
                disWts := GNNI.GetWeights(discriminator3);

                //Set obtained weights to combined
                combined1 := GNNI.SetWeights(loopCombined, disWts);

                //Train generator using combined model
                combined2 := GNNI.Fit(combined1, train_noise2, valid, batchSize, 1);

                //Get new weights to return
                comWts := GNNI.GetWeights(combined2);

                //Ignores the Discriminator part of combined fit and puts back old weights from discriminator. (Hopes to simulate discriminator.trainable = False)
                newWts := SORT(comWts(wi <= 20) + disWts(wi > 20), wi, sliceId, LOCAL);

                //gen_loss := IF(EXISTS(newWts), GNNI.GetLoss(generator), 0);
                //dis_loss := IF(EXISTS(newWts), GNNI.GetLoss(discriminator_fooled), 0);
                
                logProgress := Syslog.addWorkunitInformation('GAN training - Epoch : '+epochNum
                //+' Generator loss : '+gen_loss+' Discriminator loss : '+dis_loss
                );

                actions := ORDERED(discriminator1, combined1, discriminator2, discriminator3, combined2, logProgress);

                RETURN WHEN(newWts, actions, BEFORE);
        END;        

        finalWts := LOOP(wts, ROUNDUP(numEpochs), train(ROWS(LEFT),COUNTER));
        //finalWts := train(wts, 1);

        //Final model IDs
        finalGenerator := generator + numEpochs + 1;

        //OUTPUT actions
        //dis_out := OUTPUT(finalDiscriminator, NAMED('discriminator_id'));
        //com_out := OUTPUT(finalCombined, NAMED('combined_id'));

        //Setting new weights
        generator_trained := GNNI.SetWeights(finalGenerator, finalWts);

        RETURN generator_trained;
END;        

generator := GAN_train(trainX);

generated := GNNI.Predict(generator, train_noise);

generated_data := Tensor.R4.GetData(generated);

OUTPUT(generated_data);