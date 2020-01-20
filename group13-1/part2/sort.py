from __future__ import print_function
import sys
from pyspark.sql import SparkSession


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: submit-spark sort.py <input_file> <output_file>")
        exit(-1)

    spark = SparkSession.builder.appName("DatasetSort").getOrCreate()

    file_loc = sys.argv[1]
    res_loc = sys.argv[2]
    df = spark.read.format("CSV").option("header","true").load(file_loc)
    res = df.orderBy('cca2', 'timestamp')
    res.write.csv(path=res_loc, header="true")
    spark.stop() 
