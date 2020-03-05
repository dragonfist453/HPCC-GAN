IMPORT Python3 AS Python;
//IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Thorlib;
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
                            SELF.indexes := [LEFT.id, (COUNTER-1)%28+1, (COUNTER-1) DIV 28+1, 1],
                            SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 )); 

//Tensor dataset of 1s
valid := DATASET(batchSize, TRANSFORM(TensData,
                        SELF.indexes := [COUNTER, 1],
                        SELF.value := 1));

//Tensor dataset of 0s
fake := DATASET(batchSize, TRANSFORM(TensData,
                        SELF.indexes := [COUNTER, 1],
                        SELF.value := 0));

//Random set of normal data
/*random_data := DATASET(batchSize*latentDim, TRANSFORM(TensData,
                        SELF.indexes := [COUNTER, (COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1, 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1)); */

random_data := DATASET(latentDim, TRANSFORM(TensData,
                        SELF.indexes := [1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));

//whatisthis := SET(random_yeah, value);

//OUTPUT(whatisthis);

//OUTPUT(random_data, NAMED('whatever'));

//Builds tensors for the neural network
trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], trainX0); 
train_valid := Tensor.R4.MakeTensor([0,1],valid);
train_fake := Tensor.R4.MakeTensor([0,1],fake);
noise := Tensor.R4.MakeTensor([0,latentDim], random_data);


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
                        {'img','''layers.Reshape((28,28,1))''',['g10']},                //Generate output
                        //{'input_d','''layers.Input(shape=(28,28,1))''',['img']},        //Input of image from Generator
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

generator := GNNI.DefineModel(s, ldef_generator, compiledef_generator); //Generator model definition

discriminator := GNNI.DefineModel(s, ldef_discriminator, compiledef_discriminator); //Discriminator model definition

//combined := GNNI.DefineFuncModel(s, fldef_combined, ['noise'],['validity'],compiledef_combined); //Combined model definition

//Prerequisite for sequential training
maxX_work_item := MAX(trainX,wi);
adjY := PROJECT(train_valid, TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi + maxX_work_item, SELF := LEFT));
xyData := trainX + adjY;

//loopFunc(DATASET(t_Tensor) xyData, UNSIGNED ctr) := FUNCTION

        trainX_cur := xyData(wi <= maxX_work_item); // Recover X data
        train_valid_cur := PROJECT(xyData(wi > maxX_work_item), TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi - maxX_work_item, SELF := LEFT));// Recover Y data

        gen_imgs := GNNI.Predict(generator, noise); //Generate images from noise

        //This has problems as we need to fit a random batch and not fit the whole dataset for GAN. This will cause overfitting which is BAD. 

        discriminator_real := GNNI.Fit(discriminator, trainX_cur, train_valid_cur, numEpochs := 1); //Fit real
        discriminator_fake := GNNI.Fit(discriminator_real, gen_imgs, train_fake, numEpochs := 1); //Fit fake

        generator_net := GNNI.Fit(generator, noise, train_valid_cur, numEpochs := 1); //Fit generator and make it believe that it's noise is valid

        //RETURN true;
//END;

//numIterations := 1;
//finalXY := LOOP(xyData, numIterations, loopFunc(ROWS(LEFT), COUNTER));

gen_imgs1 := GNNI.Predict(generator_net, noise); //Just to test if all dimensions are correct and if it predicts without any training

gen_imgs2 := GNNI.Predict(discriminator_fake, gen_imgs1); //Just to test if all dimensions are correct and if it predicts without any training

gen_data := Tensor.R4.GetData(gen_imgs2);

gen_data1 := Tensor.R4.GetData(gen_imgs1);

img_data := NORMALIZE(gen_data1, 1, TRANSFORM(IMG_FORMAT,
                        SELF.id := LEFT.indexes[1]*LEFT.indexes[2]*LEFT.indexes[3],
                        SELF.image := (>DATA<) (UNSIGNED1) ((REAL) LEFT.value*127.5 + 1)
                        ));

OUTPUT(img_data, NAMED('gen'));                        
OUTPUT(gen_data, NAMED('diss'));
     

/*
//How to loop for iterative training of GANs
// Combine X and Y into one dataset
maxX_work_item := MAX(trainX);
adjY := PROJECT(valid, TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi + maxX_work_item, SELF := LEFT));
xyData := trainX + adjY;

// Loop function
loopFunc(DATASET(t_Tensor) xyData, UNSIGNED ctr) := FUNCTION
trainX := xyData(wi <= maxX_work_item) // Recover X data
// Recover Y data
valid := PROJECT(xyData(wi > maxX_work_item, TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi - maxX_work_item, SELF := LEFT));
modelId := GNNI.Fit(, trainX, valid, numEpochs = 1);
newYDat := Predict()
// Now merge predictions back into X or Y data
...
// Now re-combine X and Y
...
END;

finalXY := LOOP(xyData, numIterations, loopFunc(ROWS(LEFT), COUNTER));
*/


/*
Useful for visualising the output when it's there
img_data := NORMALIZE(gen_data1, 1, TRANSFORM(IMG_FORMAT,
                        SELF.id := LEFT.indexes[1]*LEFT.indexes[2]*LEFT.indexes[3],
                        SELF.image := (>DATA<) (UNSIGNED1) ((REAL) LEFT.value*127.5 + 1)
                        ));
*/