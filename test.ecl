/*
Basically a dump file for extra codes which could be used and also for testing out stuff along with test.ecl
*/
IMPORT Python3 as Python;
IMPORT GNN.GNNI;
IMPORT GNN.Tensor;
IMPORT GNN.Types;
IMPORT IMG.IMG;
IMPORT GNN.Internal as int;
TensData := Tensor.R4.TensData;
t_Tensor := Tensor.R4.t_Tensor;
FuncLayerDef := Types.FuncLayerDef;
RAND_MAX := POWER(2,32) - 1;
#option('outputLimit',200);

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
batchSize := 100;

//Take MNIST dataset using IMG module
mnist_train_images := IMG.MNIST_train_image();

//Tensor dataset having image data normalised to range of -1 to 1
trainX0 := NORMALIZE(mnist_train_images, imgSize, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28+1, (COUNTER-1)%28+1, 1],
                            SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 ));

trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], choosen(trainX0,1));   

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


s := GNNI.GetSession();

generator := GNNI.DefineModel(s, ldef_generator, compiledef_generator); //Generator model definition

discriminator := GNNI.DefineModel(s, ldef_discriminator, compiledef_discriminator); //Discriminator model definition

combined := GNNI.DefineFuncModel(s, fldef_combined, ['noise','img'],['img','validity'],compiledef_combined); //Combined model definition

//Noise for generator to make fakes
random_data1 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
        SELF.indexes := [(COUNTER-1) DIV batchSize + 1, (COUNTER-1)%latentDim + 1],
        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
noise := Tensor.R4.MakeTensor([batchSize,latentDim], random_data1);

gen_imgs1 := GNNI.Predict(generator,noise); //Just to test if all dimensions are correct and if it predicts without any training

OUTPUT(gen_imgs1, ,'~test::whatchamacallit',OVERWRITE);

umm := DATASET('~test::whatchamacallit', t_Tensor, FLAT);

gen_imgs2 := GNNI.Predict(discriminator, umm); //Just to test if all dimensions are correct and if it predicts without any training

gen_data := Tensor.R4.GetData(gen_imgs2);

gen_data1 := Tensor.R4.GetData(gen_imgs1);

gen_imgs := GNNI.Predict(combined, noise);

gen_data2 := Tensor.R4.GetData(gen_imgs);

OUTPUT(gen_data1, NAMED('gen'));   
OUTPUT(gen_data, NAMED('diss'));                     
//OUTPUT(gen_data2, NAMED('comb'));