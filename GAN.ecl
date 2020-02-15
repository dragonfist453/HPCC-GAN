IMPORT Python3 AS Python;
//IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Thorlib;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

RAND_MAX := POWER(2,32) - 1;
#option('outputLimit',200);

//Input definition of records

// Test parameters
trainCount := 1000;
featureCount := 5;
// END Test parameters

// Prepare training data.
// We use 5 inputs (X) and a single output (Y)
trainRec := RECORD
  UNSIGNED8 id;
  SET OF REAL x;
END;

// Build the training data.  Pick random data for X values, and use a polynomial
// function of X to compute Y.
train0 := DATASET(trainCount, TRANSFORM(trainRec,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5],)
                      );

// Break the training and test data into X (independent) and Y (dependent) data sets.  Format as Tensor Data.
trainX0 := NORMALIZE(train0, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));

// Form a Tensor from the tensor data.  This packs the data into 'slices' that can contain dense
// or sparse portions of the Tensor.  If the tensor is small, it will fit into a single slice.
// Huge tensors may require many slices.  The slices also contain tensor metadata such as the shape.
// For record oriented data, the first component of the shape should be 0, indicating that it is an
// arbitrary length set of records.
trainX := Tensor.R4.MakeTensor([0, featureCount], trainX0);

//Layer definition of models
ldef_generator := ['''layers.LeakyReLU(alpha=0.2)''',
                    '''layers.BatchNormalization(momentum=0.8)''',
                    '''layers.Dense(512)''',
                    '''layers.LeakyReLU(alpha=0.2)''',
                    '''layers.BatchNormalization(momentum=0.8)''',
                    '''layers.Dense(1024)''',
                    '''layers.LeakyReLU(alpha=0.2)''',
                    '''layers.BatchNormalization(momentum=0.8)''',
                    '''layers.Dense(featureCount,activation='tanh')'''];
                    
compiledef_generator := '''compile(loss='mse', optimizer=optimizer)''';                     

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

ldef_discriminator := ['''layers.Dense(512,input_dim=featureCount)''',
                        '''layers.LeakyReLU(alpha=0.2)''',
                        '''layers.Dense(256)''',
                        '''layers.LeakyReLU(alpha=0.2)''',
                        '''layers.Dense(1,activation='sigmoid')'''];

compiledef_discriminator := '''compile(loss='mse',
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

//Start session for GAN
s := GNNI.GetSession();

//Generator model definitiion
generator := GNNI.DefineModel(s,ldef_generator,compiledef_generator);

//Discriminator model definition
discriminator := GNNI.DefineModel(s,ldef_generator,compiledef_generator);

//Make record for y
outputRec := RECORD
        UNSIGNED8 id;
        UNSIGNED8 num;
END;

batchsize := 32;

//Give dataset with one constant (np.ones or np.zeros)
outputRec fillwith(UNSIGNED8 i,UNSIGNED8 x) := TRANSFORM
        SELF.id := i;
        SELF.num := x;
END;

//Generate ones and zeros
valid := DATASET(batchsize,fillwith(COUNTER,1));
fake := DATASET(batchsize,fillwith(COUNTER,0));

OUTPUT(valid, NAMED('valid'));
OUTPUT(fake,NAMED('fake'));

valid_ten := NORMALIZE(valid, 1, TRANSFORM(TensData,
                        SELF.indexes := [LEFT.id,COUNTER],
                        SELF.value := LEFT.num));

valid_tensor := Tensor.R4.MakeTensor([0, 2], valid_ten);

fake_ten := NORMALIZE(fake, 1, TRANSFORM(TensData,
                        SELF.indexes := [LEFT.id,COUNTER],
                        SELF.value := LEFT.num));

fake_tensor := Tensor.R4.MakeTensor([0, 2], fake_ten);

random_set := DATASET(trainCount, TRANSFORM(trainRec,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5],)
                      );

// Break the training and test data into X (independent) and Y (dependent) data sets.  Format as Tensor Data.
rand := NORMALIZE(random_set, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));

discriminator_model:= GNNI.Fit(discriminator, trainX,valid_tensor,batchSize := 128, numEpochs := 5);

