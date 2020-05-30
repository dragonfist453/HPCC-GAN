IMPORT Python3 AS Python;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Thorlib;
IMPORT Std.System.Log AS Syslog;
IMPORT STD;
IMPORT IMG.IMG;
IMPORT GNN.Utils;
nNodes := Thorlib.nodes();
nodeId := Thorlib.node();
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;
FuncLayerDef := Types.FuncLayerDef;

RAND_MAX := POWER(2,8) - 1;
RAND_MAX_2 := RAND_MAX / 2;
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
batchSize := 103;
numEpochs := 100;
epochNum := 1;
outputRows := 5;
outputCols := 5;

//Despray variables
serv := 'server=http://172.16.2.240:8010 ';
over := 'overwrite=1 ';
action  := 'action=despray ';
dstip   := 'dstip=172.16.2.240 ';
dstfile := 'dstfile=/var/lib/HPCCSystems/mydropzone/*.png ';
srcname := 'srcname=~gan::output_image ';
splitprefix := 'splitprefix=filename,filesize ';
cmd := serv + over + action + dstip + dstfile + srcname + splitprefix;

//Take MNIST dataset using IMG module
mnist_train_images := IMG.MNIST_train_image();

//Tensor dataset having image data normalised to range of -1 to 1
trainX0 := NORMALIZE(choosen(mnist_train_images,2000), imgSize, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28+1, (COUNTER-1)%28+1, 1],
                            SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 ));
OUTPUT(trainX0, NAMED('Train_TensData'));

//Random set of normal data
random_data := DATASET(latentDim*outputRows*outputCols, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / RAND_MAX)));
OUTPUT(random_data, NAMED('Noise_TensData'));
                     
//Builds tensors for the neural network
trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], trainX0);
OUTPUT(trainX, NAMED('Train_tensor')); 
train_noise := Tensor.R4.MakeTensor([0,latentDim], random_data);
OUTPUT(train_noise, NAMED('Noise_tensor'));

//Secret item for later ;)
recordCount := TENSOR.R4.GetRecordCount(trainX);
OUTPUT(recordCount, NAMED('TrainRecordCount'));

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
OUTPUT(generator, NAMED('generator_id'));

//This is used to extract weights from combined and also merge weights back
gen_wts_id := MAX(GNNI.GetWeights(generator), wi);


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
OUTPUT(discriminator, NAMED('discriminator_id'));


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
                '''layers.Reshape((28,28,1))''', //20
                '''layers.Flatten(input_shape=(28,28,1), trainable=False)''',//1
                '''layers.Dense(512,trainable=False)''',//2
                '''layers.LeakyReLU(alpha=0.2, trainable=False)''',//3
                '''layers.Dense(256,trainable=False)''',//4
                '''layers.LeakyReLU(alpha=0.2, trainable=False)''',//5
                '''layers.Dense(1,activation='sigmoid',trainable=False)'''];//6

compiledef_combined := '''compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';

//Define combined functional network
combined := GNNI.DefineModel(session, ldef_combined, compiledef_combined);
OUTPUT(combined, NAMED('combined_id'));

//Dataset of 1s for classification
valid_data := DATASET(batchSize*nNodes, TRANSFORM(TensData,
                SELF.indexes := [COUNTER, 1],
                SELF.value := 1));
OUTPUT(valid_data, NAMED('valid_tensdata'));
valid := Tensor.R4.MakeTensor([0,1],valid_data);
OUTPUT(valid, NAMED('valid_tensor'));

//Kindly note: 0.00000001 was used instead of 0 as 0 wasn't read as a tensor data in the backend. It fits fine even in python, so it's used.
//Dataset of 0s for classification
fake_data := DATASET(batchSize*nNodes, TRANSFORM(TensData,
                SELF.indexes := [COUNTER, 1],
                SELF.value := 0.00000001));
OUTPUT(fake_data, NAMED('fake_tensdata'));
fake := Tensor.R4.MakeTensor([0,1],fake_data);
OUTPUT(fake, NAMED('fake_tensor'));

//Mixed tensor of above 2
mixed_data := DATASET(batchSize*2*nNodes, TRANSFORM(TensData,
                SELF.indexes := [COUNTER,1],
                SELF.value := IF(COUNTER <= batchSize*nNodes,1,0.00000001);
            ));
OUTPUT(mixed_data, NAMED('mixed_tensdata'));            
mixed := Tensor.R4.MakeTensor([0,1], mixed_data);
OUTPUT(mixed, NAMED('mixed_tensor_ytrain'));
Y_train := mixed;

//Get only initial combined weights
wts := GNNI.GetWeights(combined);
OUTPUT(wts, NAMED('comwts'));

DATASET(TensData) makeRandom(UNSIGNED a) := FUNCTION
    reslt := DATASET(latentDim*batchSize*nNodes, TRANSFORM(TensData,
                SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
                SELF.value := ((RANDOM() % RAND_MAX) / RAND_MAX_2) - 1 * a / a));
    RETURN reslt;
END;

//Selecting random batch of images
//Random position in Tensor which is (batchSize) less than COUNT(input)
batchPos := RANDOM()%(recordCount/nNodes - batchSize);

//Extract (batchSize) tensors starting from a random batchPos from the tensor input. Now we have a random input images of (batchSize) rows.
X_dat := int.TensExtract(trainX, batchPos, batchSize);
OUTPUT(X_dat, NAMED('extracted_trainX'));

//Noise for generator to make fakes
random_data1 := makeRandom(1);
OUTPUT(random_data1, NAMED('random1_tensdata_togenerate'));    
train_noise1 := Tensor.R4.MakeTensor([0,latentDim], random_data1);
OUTPUT(train_noise1, NAMED('noisetensor_togenerate'));

//New model IDs
loopDiscriminator := discriminator + 3*(epochNum - 1);
loopCombined := combined + 2*(epochNum - 1);
loopGenerator := generator + (epochNum - 1);

//Split weights accordingly. Generator layer <= gen_wts_id. Discriminator layers > gen_wts_id. Discriminator must be subtracted by gen_wts_id to get its proper weights
genWts := SORT(wts(wi <= (Tensor.t_WorkItem) gen_wts_id), wi, sliceid, LOCAL);
OUTPUT(genWts, NAMED('genWts_fromcom'));
splitdisWts := SORT(wts(wi > (Tensor.t_WorkItem) gen_wts_id), wi, sliceid, LOCAL);
diswts := PROJECT(splitdisWts, TRANSFORM(t_Tensor,
                    SELF.wi := LEFT.wi - gen_wts_id,
                    SELF := LEFT
                    ));
OUTPUT(diswts, NAMED('diswts_fromcom'));                    

//Setting generator weights
generator1 := GNNI.SetWeights(loopGenerator, genWts);

//Predicting using Generator for fake images
gen_X_dat1 := GNNI.Predict(generator1, train_noise1);
OUTPUT(gen_X_dat1, NAMED('generated_tensor'));

//Setting discriminator weights
discriminator1 := GNNI.SetWeights(loopDiscriminator, disWts); 

//Fitting real data
discriminator2 := GNNI.Fit(discriminator1, X_dat, valid, batchSize, 1);

//Project generated data to get 0 in first shape component
generated_dat := PROJECT(gen_X_dat1, TRANSFORM(t_Tensor,
    SELF.shape := [0] + LEFT.shape[2..],
    SELF := LEFT
    ));
OUTPUT(generated_dat, NAMED('transformed_generated_tensor'));

//Fitting generated data
discriminator3 := GNNI.Fit(discriminator2, generated_dat, fake, batchSize, 1);

//Noise for generator to make fakes
random_data2 := makeRandom(2);
OUTPUT(random_data1, NAMED('random2_tensdata_tofit'));    
train_noise2 := Tensor.R4.MakeTensor([0,latentDim], random_data2);
OUTPUT(train_noise1, NAMED('noisetensor_tofit'));

//Get discriminator weights, add 20 to it, change discriminator weights of combined model, set combined weights
updateddisWts := GNNI.GetWeights(discriminator3);
OUTPUT(updateddisWts, NAMED('fit_diswts'));
newdisWts := PROJECT(updateddisWts, TRANSFORM(t_Tensor,
                    SELF.wi := LEFT.wi + gen_wts_id,
                    SELF := LEFT
                    ));
comWts := SORT(wts(wi <= (Tensor.t_WorkItem) gen_wts_id) + newdisWts(wi > (Tensor.t_WorkItem) gen_wts_id), wi, sliceid, LOCAL);
OUTPUT(comWts, NAMED('com_wts_updated'));
combined1 := GNNI.SetWeights(loopCombined, comWts);

//Fit combined model
combined2 := GNNI.Fit(combined1, train_noise2, valid, batchSize, 1);

//Get combined weights to return
newWts := GNNI.GetWeights(combined2);
OUTPUT(newWts, NAMED('fit_comwts'));

//Final model IDs
finalGenerator := generator + numEpochs + 1;

//Setting new weights
genWts2 := SORT(wts(wi<=gen_wts_id), wi, sliceid, LOCAL);
generator_trained := GNNI.SetWeights(finalGenerator, genWts2);   

//Predict an image from noise
generated := GNNI.Predict(generator_trained, train_noise);                       

//Get the generated images 
generated_data2 := Tensor.R4.GetData(generated);
OUTPUT(generated_data2, NAMED('transformed_generator_output_tensdata'));

//Convert from tensor data to images
outputImage := IMG.TenstoImg(generated_data2);

//Convert image data to jpg format to despray
mnistgrid := IMG.OutputGrid(outputImage, outputRows, outputCols, numEpochs);

OUTPUT(mnistgrid, ,'~GAN::output_image', OVERWRITE);

//Despraying image onto landing zone
despray_image := STD.File.DfuPlusExec(cmd);
WHEN(EXISTS(mnistgrid), despray_image);