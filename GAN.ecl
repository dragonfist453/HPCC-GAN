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
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28+1, (COUNTER-1)%28+1, 1],
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
                    '''layers.Reshape((28,28,1))'''];
                    
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

/*
Combined model is Generator + Discriminator
Input1 --> noise
Output1 --> image
Input2 --> image
Output2 --> validity            
*/            

//Start session for GAN
//s1 := GNNI.GetSession();

//generator := GNNI.DefineModel(s1, ldef_generator, compiledef_generator); //Generator model definition

//s2 := GNNI.GetSession();

//discriminator := GNNI.DefineModel(s2, ldef_discriminator, compiledef_discriminator); //Discriminator model definition

//combined := GNNI.DefineFuncModel(s2, fldef_combined, ['noise'],['validity'],compiledef_combined); //Combined model definition

//gen_imgs := GNNI.Predict(combined, noise); //Just to test if all dimensions are correct and if it predicts without any training

//gen_data := Tensor.R4.GetData(gen_imgs);

//OUTPUT(gen_data, ,'~GAN::deleteafterseeing',OVERWRITE);

OK := RECORD
        UNSIGNED8 id;
        UNSIGNED8 value;
END;

random_index := DATASET(batchSize, TRANSFORM(OK,
                                SELF.id := COUNTER;
                                SELF.value := RANDOM() % 60000));

OUTPUT(random_index);       

//Things to do here
//In a loop, extract samples from trainX tensor using above transform, for every batch
//Give combined both generated and actual by passing noise and getting 1 output img from generator
//Train it for how many ever epochs using the looping algorithm given 
//Optimise and remove lines. Make storage. Keep stuff to show 
//Output generated images after GAN trains as images. Add that output mechanism to IMG module
//That's it for now tbh