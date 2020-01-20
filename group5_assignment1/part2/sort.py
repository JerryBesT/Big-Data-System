from __future__ import print_function
import sys
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: submit-spark sort.py <input_dir> <output_dir>")
        exit(-1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Create a spark config
    conf = SparkConf()
    conf.set("spark.executor.cores", "5")
    conf.set("spark.driver.memory", "24g")
    conf.set("spark.executor.memory", "24g")
    conf.set("spark.task.cpus", "1")

    # Initialize spark session with the config
    spark = SparkSession.builder.appName("ExportSort").config(conf=conf).getOrCreate()

    # Read data from input file into Spark DataFrame
    data = spark.read.format("CSV").option("header", "true").load(input_dir)

    # Sort data according to cca2 and timestamp
    out = data.orderBy('cca2', 'timestamp')

    # Write the output file
    out.write.csv(path=output_dir, header="true")
    spark.stop()
