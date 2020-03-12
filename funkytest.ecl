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

/*
Combined model is Generator + Discriminator
Input1 --> noise
Output1 --> image
Input2 --> image
Output2 --> validity            
*/            

s := GNNI.GetSession();

combined := GNNI.DefineFuncModel(s, fldef_combined, ['noise'],['validity'],compiledef_combined); //Combined model definition

result := GNNI.Predict(combined, train_noise);

result_data := Tensor.R4.GetData(result);

OUTPUT(result_data);