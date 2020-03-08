IMPORT Python3 AS Python;
//IMPORT $.^ AS GNN;
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

//Input data 

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

//OUTPUT(mnist_train_images, ,'~image_db::mnist_train_images',OVERWRITE);

//Tensor dataset having image data normalised to range of -1 to 1
trainX0 := NORMALIZE(mnist_train_images, imgSize, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28+1, (COUNTER-1)%28+1, 1],
                            SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 )); 

/*
//Tensor dataset of 1s
valid := DATASET(batchSize, TRANSFORM(TensData,
                        SELF.indexes := [COUNTER, 1],
                        SELF.value := 1));

//Tensor dataset of 0s
fake := DATASET(batchSize, TRANSFORM(TensData,
                        SELF.indexes := [COUNTER, 1],
                        SELF.value := 0));
*/


//Random set of normal data
random_data := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV batchSize + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                     

//OUTPUT(random_data, NAMED('whatever'));

//Builds tensors for the neural network
trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], trainX0); 
//train_valid := Tensor.R4.MakeTensor([0,batchSize],valid);
//train_fake := Tensor.R4.MakeTensor([0,batchSize],fake);
train_noise := Tensor.R4.MakeTensor([batchSize,latentDim], random_data);


//Layer definition of models
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

/*
Generator model in keras
Layers:-
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(credit_size, activation='tanh'))
Compile specs:-
        compile(loss='binary_crossentropy', optimizer=optimizer)
*/

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
   
/*
Discriminator model in keras
Layers:-
        model.add(Dense(512,input_dim=credit_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
Compile specs:- 
         compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])         
*/

/*
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
*/

/*
Combined model is Generator + Discriminator
Input1 --> noise
Output1 --> image
Input2 --> image
Output2 --> validity            
*/            

//Start session for GAN
s := GNNI.GetSession();

//generator := GNNI.DefineModel(s, ldef_generator, compiledef_generator); //Generator model definition

//discriminator := GNNI.DefineModel(s, ldef_discriminator, compiledef_discriminator); //Discriminator model definition

//combined := GNNI.DefineFuncModel(s, fldef_combined, ['noise'],['img'],compiledef_combined); //Combined model definition

UNSIGNED4 GAN_train(UNSIGNED4 session,
                        DATASET(t_Tensor) input,
                        UNSIGNED4 batchSize = 100,
                        UNSIGNED4 numEpochs = 1) := FUNCTION

        //Secret item for later ;)
        numData := COUNT(input);

        //Define generator network
        generator := GNNI.DefineModel(session, ldef_generator, compiledef_generator); //Generator model definition

        //Define discriminator network
        discriminator := GNNI.DefineModel(session, ldef_discriminator, compiledef_discriminator); //Discriminator model definition

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

        wts1 := GNNI.GetWeights(generator);

        DATASET(t_Tensor) train(DATASET(t_Tensor) wts, UNSIGNED4 epochNum) := FUNCTION
                //Random position in Tensor which is (batchSize) less than COUNT(input)
                batchPos := RANDOM()%numData - batchSize;

                //Extract (batchSize) tensors starting from a random batchPos from the tensor input. Now we have a random input images of (batchSize) rows.
                X_dat := int.TensExtract(input, batchPos, batchSize);

                //Noise for generator to make fakes
                random_data1 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV batchSize + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                train_noise1 := Tensor.R4.MakeTensor([batchSize,latentDim], random_data1);        

                //Fitting real data
                discriminator_real := GNNI.Fit(discriminator, X_dat, valid, numEpochs, batchSize);

                //Predicting using Generator for fake images
                gen_X_dat1 := GNNI.Predict(generator, train_noise1);
                
                //Fitting random data
                discriminator_fake := GNNI.Fit(discriminator_real, gen_X_dat1, fake, numEpochs, batchSize);

                //Noise for generator to make fakes
                random_data2 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV batchSize + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
                train_noise2 := Tensor.R4.MakeTensor([batchSize,latentDim], random_data2);

                //Predicting using Generator for fake images
                gen_X_dat2 := GNNI.Predict(generator, train_noise2);

                //Fitting real data
                discriminator_fooled := GNNI.Fit(discriminator_fake, gen_X_dat2, valid, numEpochs, batchSize);

                gen_loss := GNNI.GetLoss(generator);
                dis_loss := GNNI.GetLoss(discriminator_fooled);
                newWts := GNNI.GetWeights(generator);

                logProgress := Syslog.addWorkunitInformation('GAN training - Epoch : '+epochNum+' Generator loss : '+gen_loss+' Discriminator loss : '+dis_loss);
                RETURN WHEN(newWts, logProgress);
        END;        

        finalWts := LOOP(wts1, ROUNDUP(numEpochs), train(ROWS(LEFT),COUNTER));

        generator_wts := GNNI.SetWeights(generator, finalWts);

        RETURN generator_wts;
END;        

final := GAN_train(s, trainX);

generated := GNNI.Predict(final, train_noise);

generated_data := Tensor.R4.GetData(generated);

OUTPUT(generated_data);

//gen_imgs1 := GNNI.Predict(generator,noise); //Just to test if all dimensions are correct and if it predicts without any training

//gen_imgs2 := GNNI.Predict(discriminator, gen_imgs1); //Just to test if all dimensions are correct and if it predicts without any training

//gen_data := Tensor.R4.GetData(gen_imgs2);

//gen_data1 := Tensor.R4.GetData(gen_imgs1);

//gen_imgs := GNNI.Predict(combined, train_noise);

//gen_data2 := Tensor.R4.GetData(gen_imgs);

//OUTPUT(gen_data, NAMED('diss'));
//OUTPUT(gen_data1, NAMED('gen'));                        
//OUTPUT(gen_data2, NAMED('comb'));


/*
Useful for visualising the output when it's there
img_data := NORMALIZE(gen_data1, 1, TRANSFORM(IMG_FORMAT,
                        SELF.id := LEFT.indexes[1]*LEFT.indexes[2]*LEFT.indexes[3],
                        SELF.image := (>DATA<) (UNSIGNED1) ((REAL) LEFT.value*127.5 + 1)
                        ));
*/