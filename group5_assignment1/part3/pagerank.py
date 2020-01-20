from __future__ import print_function
import sys
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import re

def preprocess(line):
    if line.startswith('#'):
        return False
    line = line.strip().split('\t')
    if len(line) != 2:
        return False
    for item in line:
        if item.startswith('Category:'):
            continue
        if ':' in item:
            return False
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: pagerank <input dir> <output dir>")
        exit(-1)

    # Set Spark driver memory to 24GB and executor memory to 24GB. 
    # Set executor cores to be 5 and number of cpus per task to be 1
    conf = SparkConf()
    conf.set("spark.executor.cores", "5")
    conf.set("spark.driver.memory", "26g")
    conf.set("spark.executor.memory", "26g")
    conf.set("spark.task.cpus", "1")

    spark = SparkSession.builder.appName("PageRank").config(conf=conf).getOrCreate()

    '''
    Example input: url1 url2
    get rid of the comments
    get rid of the unwanted sign ":", which may be in the urls
    convert all the urls to be lower case
    '''
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0]) \
        .filter(lambda k: preprocess(k)) \
        .map(lambda k: k.lower())

    '''
    Example input: url1 url2
    parse them into tuple (url1, url2)
    filter them so that no duplicate tuples
    group the according to url1 in the tuple
    hash the tuples into 15 partitions
    '''
    links = lines.map(lambda line: tuple(line.strip().split('\t'))).distinct().groupByKey() \
        .partitionBy(15, lambda k: hash(k)).cache()

    # Assign initial ranks to all the tuples
    ranks = links.map(lambda urls: (urls[0], 1.0))

    # Iterate the PageRank algorithm for 10 times
    for i in range(10):
        # Update contribs and ranks
        contribs = links.join(ranks).flatMap(lambda url_rank: [(url, url_rank[1][1] / len(url_rank[1][0])) for url in url_rank[1][0]])
        ranks = contribs.reduceByKey(lambda a, b: a + b).mapValues(lambda s: 0.15 + s * 0.85)

    # Convert ranks rdd into DataFrame and save file
    ranks = ranks.toDF()
    ranks.write.text(path=sys.argv[2])
    spark.stop()
