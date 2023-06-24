from __future__ import print_function

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd
import sys

def to_spark_df(fin):
    df = pd.read_csv(fin)
    df.fillna("", inplace=True)
    df = spark.createDataFrame(df)
    return(df)

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ToxicCommentClassification")\
        .getOrCreate()

    train = to_spark_df(sys.argv[1])
    test = to_spark_df(sys.argv[2])

    out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]

    tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
    wordsData = tokenizer.transform(train)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    tf = hashingTF.transform(wordsData)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(tf) 
    tfidf = idfModel.transform(tf)

    REG = 0.1

    test_tokens = tokenizer.transform(test)
    test_tf = hashingTF.transform(test_tokens)
    test_tfidf = idfModel.transform(test_tf)
    test_res = test.select('id')

    extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())

    for col in out_cols:
        lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=REG)
        lrModel = lr.fit(tfidf)
        res = lrModel.transform(test_tfidf)
        test_res = test_res.join(res.select('id', 'probability'), on="id")
        test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")

    test_res.show()

    spark.stop()




