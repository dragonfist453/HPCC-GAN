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

//Format of labels
LABEL_FORMAT := RECORD
    UNSIGNED id;
    DATA1 label;
END;


//Train data definitions
imgcount_train := 60000;
imgRows := 28;
imgCols := 28;
imgChannels := 1;
imgSize := imgRows * imgCols;
latentDim := 100;
numClasses := 10;

mnist_train_images := IMG.MNIST_train_image(); //Returns MNIST training images record from IMG module

//making sure that IMG works
OUTPUT(mnist_train_images, ,'~thor::mnist_train_images',OVERWRITE); //works

mnist_train_labels := IMG.MNIST_train_label(); //Returns MNIST training labels record from IMG module

//making sure that IMG labels works too
OUTPUT(mnist_train_labels, ,'~thor::mnist_train_labels',OVERWRITE); //works


trainX0 := NORMALIZE(mnist_train_images, imgSize, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28+1, (COUNTER-1)%28+1, 1],
                            SELF.value := (REAL) (>UNSIGNED1<) LEFT.image[counter])); //works great. consumes lot of time as 47040000 records are made

OUTPUT(trainX0, ,'tensor_data::X_train',OVERWRITE);
 
trainY0 := NORMALIZE(mnist_train_labels, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id],
                            SELF.value := (REAL) (>UNSIGNED1<) LEFT.label)); //works great. much quicker than last one

OUTPUT(trainY0, ,'~tensor_data::Y_train',OVERWRITE);

//Call OneHotEncoder to get 10 output tensor
trainY_OH := Utils.ToOneHot(trainY0, numClasses);
OUTPUT(trainY_OH, ,'~tensor_data::Y_train_OH',OVERWRITE);

//Builds tensors for the neural network
trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], trainX0); 
trainY:= Tensor.R4.MakeTensor([0, numClasses], trainY_OH);

//OUTPUT(trainX, ,'tensor::X_train',OVERWRITE);
//OUTPUT(trainY, ,'tensor::y_train',OVERWRITE);


//Test data definitions
imgcount_test := 10000;

mnist_test_images := IMG.MNIST_test_image(); //Returns MNIST test images record from IMG module

//making sure that IMG works
OUTPUT(mnist_test_images, ,'~thor::mnist_test_images',OVERWRITE); 

mnist_test_labels := IMG.MNIST_test_label(); //Returns MNIST test labels record from IMG module

//making sure that IMG labels works too
OUTPUT(mnist_test_labels, ,'~thor::mnist_test_labels',OVERWRITE); 


testX0 := NORMALIZE(mnist_test_images, imgSize, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28 + 1, (COUNTER-1)%28 + 1, 1],
                            SELF.value := (REAL) (>UNSIGNED1<) LEFT.image[counter]));

OUTPUT(testX0, ,'~tensor_data::X_test',OVERWRITE); //very slow. 47040000 records build here (takes storage)
 
testY0 := NORMALIZE(mnist_test_labels, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id],
                            SELF.value := (REAL) (>UNSIGNED1<) LEFT.label));

OUTPUT(testY0, ,'~tensor_data::Y_test',OVERWRITE); //much faster than image tensor

//Call OneHotEncoder to get 10 output tensor
testY_OH := Utils.ToOneHot(testY0, numClasses);
OUTPUT(testY_OH, ,'~tensor_data::Y_test_OH',OVERWRITE);

//Builds tensors for the neural network
testX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], testX0);
testY:= Tensor.R4.MakeTensor([0, numClasses], testY_OH);

//OUTPUT(testX, ,'tensor::X_test',OVERWRITE);
//OUTPUT(testY, ,'tensor::y_test',OVERWRITE);


//Layer definition of models
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
                    '''layers.Dense(featureCount,activation='tanh')''',
                    '''layers.Reshape((28,28,1))'''];
                    
compiledef_generator := '''compile(loss='binary_crossentropy', optimizer=optimizer)''';                     

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

ldef_discriminator := ['''layers.Flatten(input_shape=(28,28,1))''',
                        '''layers.Dense(512,input_dim=featureCount)''',
                        '''layers.LeakyReLU(alpha=0.2)''',
                        '''layers.Dense(256)''',
                        '''layers.LeakyReLU(alpha=0.2)''',
                        '''layers.Dense(1,activation='sigmoid')'''];

compiledef_discriminator := '''compile(loss='binary_crossentropy',
                                optimizer=optimizer,
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

fldef_combined := DATASET([{'noise','''''',[]}], 
                FuncLayerDef);

/*
Ref
fldef := DATASET([{'input1', '''layers.Input(shape=(5,))''', []},  // Regression Input
                {'d1', '''layers.Dense(256, activation='tanh')''', ['input1']}, // Regression Hidden 1
                {'d2', '''layers.Dense(256, activation='relu')''', ['d1']},   // Regression Hidden 2
                {'output1', '''layers.Dense(1, activation=None)''', ['d2']}, // Regression Output
                {'input2', '''layers.Input(shape=(5,))''', []}, // Classification Input
                {'d3', '''layers.Dense(16, activation='tanh', input_shape=(5,))''',['input2']}, // Classification Hidden 1
                {'d4', '''layers.Dense(16, activation='relu')''',['d3']}, // Classification Hidden 2
                {'output2', '''layers.Dense(3, activation='softmax')''', ['d4']}], // Classification Output
            FuncLayerDef);
Input1 --> noise
Output1 --> image
Input2 --> image
Output2 --> validity            
*/            
//Start session for GAN
s := GNNI.GetSession();