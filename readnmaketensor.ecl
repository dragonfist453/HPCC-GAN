IMPORT GNN.Tensor;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;
#option('outputLimit',200);

//Input definition of records

Credit := RECORD
    UNSIGNED id;    
    UNSIGNED Time;
    REAL V1;
    REAL V2;
    REAL V3;
    REAL V4;
    REAL V5;
    REAL V6;
    REAL V7;
    REAL V8;
    REAL V9;
    REAL V10;
    REAL V11;
    REAL V12;
    REAL V13;
    REAL V14;
    REAL V15;
    REAL V16;
    REAL V17;
    REAL V18;
    REAL V19;
    REAL V20;
    REAL V21;
    REAL V22;
    REAL V23;
    REAL V24;
    REAL V25;
    REAL V26;
    REAL V27;
    REAL V28;
    REAL Amount;
    BOOLEAN Class;
END;

credit_size := 32;

fraudData := DATASET('~gan::test::creditcard.csv',Credit,CSV);

//f := fraudData(Class=true);
//nf := fraudData(Class=false);
train := NORMALIZE(fraudData, credit_size,
                 TRANSFORM(TensData,
                        SELF.indexes := [LEFT.id,Counter],
                        SELF.value := MAP(Counter = 1 => LEFT.Time,
                                        Counter = 2 => LEFT.V1,
                                        Counter = 3 => LEFT.V2,
                                        Counter = 4 => LEFT.V3,
                                        Counter = 5 => LEFT.V4,
                                        Counter = 6 => LEFT.V5,
                                        Counter = 7 => LEFT.V6,   
                                        Counter = 8 => LEFT.V7,
                                        Counter = 9 => LEFT.V8,
                                        Counter = 10 => LEFT.V9,   
                                        Counter = 11 => LEFT.V10,
                                        Counter = 12 => LEFT.V11,
                                        Counter = 13 => LEFT.V12,
                                        Counter = 14 => LEFT.V13,
                                        Counter = 15 => LEFT.V14,
                                        Counter = 16 => LEFT.V15,
                                        Counter = 17 => LEFT.V16,
                                        Counter = 18 => LEFT.V17,
                                        Counter = 19 => LEFT.V18,
                                        Counter = 20 => LEFT.V19,
                                        Counter = 21 => LEFT.V20,
                                        Counter = 22 => LEFT.V21,
                                        Counter = 23 => LEFT.V22,
                                        Counter = 24 => LEFT.V23,
                                        Counter = 25 => LEFT.V24,
                                        Counter = 26 => LEFT.V25,
                                        Counter = 27 => LEFT.V26,
                                        Counter = 28 => LEFT.V27,
                                        Counter = 29 => LEFT.V28,
                                        Counter = 30 => LEFT.Amount,
                                        Counter = 31 => LEFT.Class,)));

train_fraud := Tensor.R4.MakeTensor([0,credit_size],train);
OUTPUT('hellow');