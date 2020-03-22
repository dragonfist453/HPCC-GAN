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

trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], trainX0);   

//GENERATOR
//Generator model definition information
ldef_generator := ['''layers.Input(shape=(100,))''', 
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
                '''layers.Reshape((1,28,28,1))''']; //20
            
compiledef_generator := '''compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';

//DISCRIMINATOR
//Discriminator model definition information
ldef_discriminator := ['''layers.Input(shape=(28,28,1))''',
                        '''layers.Flatten(input_shape=(28,28,1))''',//1
                        '''layers.Dense(512)''',//2
                        '''layers.LeakyReLU(alpha=0.2)''',//3
                        '''layers.Dense(256)''',//4
                        '''layers.LeakyReLU(alpha=0.2)''',//5
                        '''layers.Dense(1,activation='sigmoid')'''];//6

compiledef_discriminator := '''compile(loss='binary_crossentropy',
                                optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                                metrics=['accuracy'])''';      

//COMBINED functional model
//Combined model definition information
fldef_combined := DATASET([{'noise','''layers.Input(shape=(100,))''',[]},              //Input of Generator 
{'g1','''layers.Dense(256, input_dim=100)''',['noise']},        //Generator layer 1 1
{'g2','''layers.LeakyReLU(alpha=0.2)''',['g1']},                //Generator layer 2 3
{'g3','''layers.BatchNormalization(momentum=0.8)''',['g2']},    //Generator layer 3 6
{'g4','''layers.Dense(512)''',['g3']},                          //Generator layer 4 7
{'g5','''layers.LeakyReLU(alpha=0.2)''',['g4']},                //Generator layer 5 9
{'g6','''layers.BatchNormalization(momentum=0.8)''',['g5']},    //Generator layer 6 12
{'g7','''layers.Dense(1024)''',['g6']},                         //Generator layer 7 13
{'g8','''layers.LeakyReLU(alpha=0.2)''',['g7']},                //Generator layer 8 15
{'g9','''layers.BatchNormalization(momentum=0.8)''',['g8']},    //Generator layer 9 18
{'g10','''layers.Dense(784,activation='tanh')''',['g9']},       //Generator layer 10 19
{'img','''layers.Reshape((1,28,28,1))''',['g10']},                //Generate output 20
{'d1','''layers.Flatten(input_shape=(28,28,1), trainable = False)''',['img']}, //Discriminator layer 1 21
{'d2','''layers.Dense(512, trainable = False)''',['d1']},   //Discriminator layer 2 22
{'d3','''layers.LeakyReLU(alpha=0.2, trainable = False)''',['d2']},                //Discriminator layer 3 23
{'d4','''layers.Dense(256, trainable = False)''',['d3']},                          //Discriminator layer 4 24
{'d5','''layers.LeakyReLU(alpha=0.2, trainable = False)''',['d4']},                //Discriminator layer 5 25
{'validity','''layers.Dense(1,activation='sigmoid', trainable = False)''',['d5']}],//Output of Discriminator, valid image or not 26
FuncLayerDef);

compiledef_combined := '''compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';                 


s := GNNI.GetSession();

generator := GNNI.DefineModel(s, ldef_generator, compiledef_generator); //Generator model definition
gen := OUTPUT(generator, NAMED('generator_id'));

discriminator := GNNI.DefineModel(s, ldef_discriminator, compiledef_discriminator); //Discriminator model definition
dis := OUTPUT(discriminator, NAMED('discriminator_id'));

combined := GNNI.DefineFuncModel(s, fldef_combined, ['noise'],['validity'],compiledef_combined); //Combined model definition
com := OUTPUT(combined, NAMED('combined_id'));

//Noise for generator to make fakes
random_data1 := DATASET(latentDim*5, TRANSFORM(TensData,
        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
noise := Tensor.R4.MakeTensor([0,latentDim], random_data1);

//Dataset of 1s for classification
valid_data := DATASET(1, TRANSFORM(TensData,
                SELF.indexes := [COUNTER, 1],
                SELF.value := 1));
valid := Tensor.R4.MakeTensor([0,1],valid_data);

//discriminator1 := GNNI.Fit(discriminator, trainX, valid);

gen_imgs1 := GNNI.Predict(generator,noise); //Just to test if all dimensions are correct and if it predicts without any training

OUTPUT(gen_imgs1, ,'~test::whatchamacallit',OVERWRITE);

umm := DATASET('~test::whatchamacallit', t_Tensor, FLAT);

gen_imgs3 := PROJECT(umm, TRANSFORM(t_Tensor,
            SELF.nodeId := LEFT.nodeId,
            SELF.wi := LEFT.wi,
            SELF.sliceid := LEFT.sliceid,
            SELF.shape := [0,LEFT.shape[2],LEFT.shape[3],LEFT.shape[4]],
            SELF.dataType := LEFT.dataType,
            SELF.maxslicesize := LEFT.maxslicesize,
            SELF.slicesize := LEFT.slicesize,
            SELF.denseData := LEFT.denseData,
            SELF.sparseData := LEFT.sparseData
            )); 

gen_imgs2 := GNNI.Predict(discriminator, gen_imgs3); //Just to test if all dimensions are correct and if it predicts without any training

gen_data := Tensor.R4.GetData(gen_imgs2); //It has issues as gen_imgs2 is of dim [1,28,28,1]. Make a small function to change this issue. 

gen_data1 := Tensor.R4.GetData(umm);

gen_imgs := GNNI.Predict(combined, noise);

gen_data2 := Tensor.R4.GetData(gen_imgs);

OUT_G := OUTPUT(gen_data1, NAMED('gen'));   
OUT_D := OUTPUT(gen_data, NAMED('diss'));                     
OUT_C := OUTPUT(gen_data2, NAMED('comb'));

SEQUENTIAL(OUT_G,OUT_D,OUT_C);

tensWts := GNNI.GetWeights(combined);

wts_out := Tensor.R4.GetData(tensWts);

OUTPUT(tensWts);

//tensum := int.TensExtract(trainX, 3, 100);

//OUTPUT(tensum);