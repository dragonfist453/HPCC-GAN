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
#option('outputLimit',2000);

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
trainX0 := NORMALIZE(CHOOSEN(mnist_train_images,250), imgSize, TRANSFORM(TensData,
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

//COMBINED
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
                '''layers.Flatten(input_shape=(28,28,1))''',//1
                '''layers.Dense(512)''',//2
                '''layers.LeakyReLU(alpha=0.2)''',//3
                '''layers.Dense(256)''',//4
                '''layers.LeakyReLU(alpha=0.2)''',//5
                '''layers.Dense(1,activation='sigmoid')'''];//6

compiledef_combined := '''compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';                 


s := GNNI.GetSession();

generator := GNNI.DefineModel(s, ldef_generator, compiledef_generator); //Generator model definition
//OUTPUT(generator, NAMED('generator_id'));

discriminator := GNNI.DefineModel(s, ldef_discriminator, compiledef_discriminator); //Discriminator model definition
//OUTPUT(discriminator, NAMED('discriminator_id'));

combined := GNNI.DefineModel(s, ldef_combined, compiledef_combined); //Combined model definition
//OUTPUT(combined, NAMED('combined_id'));

//Noise for generator to make fakes
random_data1 := DATASET(latentDim*100, TRANSFORM(TensData,
        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
noise := Tensor.R4.MakeTensor([0,latentDim], random_data1);

//Dataset of 1s for classification
valid_data := DATASET(100, TRANSFORM(TensData,
                SELF.indexes := [COUNTER, 1],
                SELF.value := 1));
valid := Tensor.R4.MakeTensor([0,1],valid_data);

fake_data := DATASET(100, TRANSFORM(TensData,
                SELF.indexes := [COUNTER, 1],
                SELF.value := 0));
fake := Tensor.R4.MakeTensor([0,1],fake_data);

gen_data := GNNI.Predict(generator, noise);

X_dat := int.TensExtract(trainX, 123, 100);

gen_imgs := PROJECT(gen_data, TRANSFORM(t_Tensor,
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
                            SELF.indexes := [COUNTER DIV (dim+1) + 1,LEFT.indexes[2],LEFT.indexes[3],LEFT.indexes[4]],
                            SELF := LEFT
                            ));

toNN := Tensor.R4.MakeTensor([0,imagerows,imagecols,imagechannels],toTensor);                          
OUTPUT(toNN);
//gen_out := Tensor.R4.AlignTensors(gen_data);
//output_data := X_dat + gen_out;
//OUTPUT(output_data);
/*
max_wi := MAX(valid, wi);
max_sid := MAX(valid, wi);
new_fake := PROJECT(fake, TRANSFORM(t_Tensor,
                            SELF.wi := max_wi,
                            SELF.sliceid := max_sid + COUNTER,
                            SELF := LEFT
                            )); */
whatever := valid + fake;
output_comb := Tensor.R4.AlignTensors(whatever);
//OUTPUT(output_comb);


//discriminator1 := GNNI.Fit(discriminator, trainX, valid);
/*
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

wts := GNNI.GetWeights(combined);

genWts := wts(wi <= 20);
splitdisWts := wts(wi > 20);
diswts := PROJECT(splitdisWts, TRANSFORM(t_Tensor,
                        SELF.wi := LEFT.wi - 20,
                        SELF := LEFT
                        ));

OUTPUT(disWts, NAMED('project_dis'));
OUTPUT(genWts, NAMED('project_gen'));

OUTPUT(GNNI.GetWeights(discriminator), NAMED('weird'));


combined1 := GNNI.SetWeights(combined, wts);
OUTPUT(combined1, NAMED('new_comid'));
tens3 := GNNI.GetWeights(combined1);
OUTPUT(tens3, NAMED('com_out'));

discriminator1 := GNNI.SetWeights(discriminator, disWts);
OUTPUT(discriminator1, NAMED('new_disid'));
tens1 := GNNI.GetWeights(discriminator1);
OUTPUT(tens1, NAMED('dis_out'));

generator1 := GNNI.SetWeights(generator, genWts);
OUTPUT(generator1, NAMED('new_genid'));
tens2 := GNNI.GetWeights(generator1);
OUTPUT(tens2, NAMED('gen_out'));

//tensum := int.TensExtract(trainX, 3, 100);

//OUTPUT(tensum); 
*/