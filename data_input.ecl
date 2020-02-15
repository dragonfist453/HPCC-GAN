IMPORT Python3 as Python;

#option('outputLimit',200);

Fraud := RECORD
    INTEGER Time;
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

fraudData := DATASET('~gan::test::creditcard.csv',Fraud,CSV);

OUTPUT(fraudData,NAMED('credit_card_fraud'));