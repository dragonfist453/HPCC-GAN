numImages := 60000;
numRows := 28;
numCols := 28;
imgSize := numRows * numCols;

// We should have been able to use INTEGER4 for the first four fields, but the ENDIAN seems to be
// backward from what we are expecting, so we use data4
MNIST_FORMAT := RECORD
    DATA4 magic;
    DATA4 numImages;
    DATA4 numRows;
    DATA4 numCols;
    DATA47040000 contents;  // 60000 * 28 * 28 gets us to the end of the file.
END;

// Read from the landing zone.  You'll probably have to adjust the file path.  Don't use dash(-) in the
// file name.  Change the file name to use underscores.   Note the escape characters (^) to indicate capital letters.
// Otherwise will convert to lower case.
mnist := DATASET('~test::mnist_train_images', MNIST_FORMAT, FLAT);

OUT_FORMAT := RECORD
    UNSIGNED id;
    DATA image;
END;

// This will create 60,000 records, each with one image.  The id field indicates the image number
outRecs0 := NORMALIZE(mnist, numImages, TRANSFORM(OUT_FORMAT,
                            SELF.image := LEFT.contents[((COUNTER-1)*imgSize+1) .. (COUNTER*imgSize)],
                            SELF.id := COUNTER));
// We distribute the records to spread them across cluster nodes (for further processing).  Since we read from the
// landing zone, the whole file originally came into 1 node as 1 record.  Now we have lots of records,
// so they get distributed randomly across the nodes.
outRecs := DISTRIBUTE(outRecs0, id);
// Output the original file contents to the work unit for visibility
OUTPUT(mnist, {magic, numImages, numRows, numCols, contents[1 .. 10]});
// Output the individual image records to a Logical File (i.e. Thor dataset) spread evenly across the nodes
OUTPUT(outRecs, ,'thor::mnist_images', OVERWRITE);
