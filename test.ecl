IMPORT Python3 as Python;
IMPORT GNN.GNNI;
IMPORT GNN.Tensor;
IMPORT GNN.Types;
IMPORT IMG.IMG;
IMPORT GNN.Internal as int;
IMPORT Std.System.Log AS Syslog;
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

trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], choosen(trainX0,1000));   

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
                {'img_out','''layers.Reshape((1,28,28,1))''',['g10']},                //Generate output
                {'img_in','''layers.Flatten(input_shape=(28,28,1))''',['img_out']}, //Input of Discriminator
                {'d1','''layers.Dense(512)''',['img_in']},   //Discriminator layer 1
                {'d2','''layers.LeakyReLU(alpha=0.2)''',['d1']},                //Discriminator layer 2
                {'d3','''layers.Dense(256)''',['d2']},                          //Discriminator layer 3
                {'d4','''layers.LeakyReLU(alpha=0.2)''',['d3']},                //Discriminator layer 4
                {'validity','''layers.Dense(1,activation='sigmoid')''',['d4']}],//Output of Discriminator, valid image or not
                FuncLayerDef);

compiledef_combined := '''compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';                 

s := GNNI.GetSession();

com_gen := GNNI.DefineFuncModel(s, fldef_combined, ['noise'],['img_out'],compiledef_combined); //Functional model of generator defined
com_dis := GNNI.DefineFuncModel(s, fldef_combined, ['img_in'],['validity'],compiledef_combined); //Functional model of discriminator defined

//Noise for generator to make fakes
random_data := DATASET(latentDim, TRANSFORM(TensData,
        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
noise := Tensor.R4.MakeTensor([0,latentDim], random_data);

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

wts := GNNI.GetWeights(com_dis);

DATASET(t_Tensor) train(DATASET(t_Tensor) wts, UNSIGNED4 epochNum) := FUNCTION
    loopGen := GNNI.SetWeights(com_gen, wts);

    loopDis := GNNI.SetWeights(com_dis, wts);

    images := int.TensExtract(trainX, RANDOM()%(60000-batchSize), batchSize);

    random_data1 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
            SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
            SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
    noise1 := Tensor.R4.MakeTensor([0,latentDim], random_data1);

    loopDis1 := GNNI.Fit(loopDis, images, valid, batchSize, 1);

    gen_imgs := GNNI.Predict(loopGen, noise1);

    loopDis2 := GNNI.Fit(loopDis1, gen_imgs, fake, batchSize, 1);

    loopwts := GNNI.GetWeights(loopDis2);

    loopGen1 := GNNI.SetWeights(loopGen, loopwts);

    random_data2 := DATASET(latentDim*batchSize, TRANSFORM(TensData,
            SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
            SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));
    noise2 := Tensor.R4.MakeTensor([0,latentDim], random_data2);

    loopGen2 := GNNI.Fit(loopGen1, noise, images);

    wts_after := GNNI.GetWeights(loopGen2);

    logProgress := Syslog.addWorkunitInformation('GAN training - Epoch : '+epochNum
                //+' Generator loss : '+gen_loss+' Discriminator loss : '+dis_loss
                );

    actions := ORDERED(loopGen, loopDis, loopDis1, loopDis2, loopGen1, loopGen2, logProgress);            

    RETURN WHEN(wts_after, actions, BEFORE);
END;

finalWts := LOOP(wts, ROUNDUP(2), train(ROWS(LEFT),COUNTER));

generator := GNNI.SetWeights(com_gen, finalWts);

gen_out := GNNI.Predict(generator, noise);

gen_dat := Tensor.R4.GetData(gen_out);

OUTPUT(gen_dat, NAMED('victory'));