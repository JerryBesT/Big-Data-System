from __future__ import print_function
import sys
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import re

def preprocess(line):
    line = re.split(r'\t', line, 1)
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
        print("Usage: pagerank <input file> <output file>", file=sys.stderr)
        exit(-1)
    conf = SparkConf()
    conf.set("spark.executor.cores", "5")
    conf.set("spark.driver.memory", "8g")
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.task.cpus", "1")
    spark = SparkSession.builder.appName("PageRank").config(conf=conf).getOrCreate()
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0]) \
	.filter(lambda k: not k.startswith('#')) \
        .filter(lambda k: preprocess(k)) \
        .map(lambda k: k.lower())

    links = lines.map(lambda urls: tuple(re.split(r'\t',urls.strip('\n'),1))) \
	.distinct().groupByKey().partitionBy(15, lambda k: hash(k))#.cache()

    ranks = links.map(lambda un: (un[0], 1.0))

    for iteration in range(10):
        contribs = links.join(ranks).flatMap(lambda urank: [(url,urank[1][1]/len(urank[1][0])) for url in urank[1][0]])
        ranks = contribs.reduceByKey(lambda a, b: a + b) \
		.mapValues(lambda s: 0.15 + s * 0.85)

    ranks = ranks.toDF()
    ranks.write.csv(path=sys.argv[2])
    spark.stop()
