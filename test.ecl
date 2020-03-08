IMPORT Python3 as Python;
IMPORT GNN.GNNI;
IMPORT GNN.Tensor;
IMPORT GNN.Types;
IMPORT IMG.IMG;
IMPORT GNN.Internal as int;
TensData := Tensor.R4.TensData;
t_Tensor := Tensor.R4.t_Tensor;
RAND_MAX := POWER(2,32) - 1;
#option('outputLimit',200);

imgcount_train := 60000;
imgRows := 28;
imgCols := 28;
imgChannels := 1;
imgSize := imgRows * imgCols;
latentDim := 100;
numClasses := 10;
batchSize := 128;

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

random_data := DATASET(batchSize*latentDim, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / (RAND_MAX/2)) -1));

train_valid := Tensor.R4.MakeTensor([0,batchSize],valid);
train_fake := Tensor.R4.MakeTensor([0,batchSize],fake);
train_noise := Tensor.R4.MakeTensor([batchSize,latentDim], random_data);

OUTPUT(train_noise);

OUTPUT(Tensor.R4.GetRecordCount(train_valid));
OUTPUT(Tensor.R4.GetRecordCount(train_fake));
OUTPUT(Tensor.R4.GetRecordCount(train_noise));                        