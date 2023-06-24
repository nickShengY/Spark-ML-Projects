from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, when
import sys

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("CensusIncomePrediction")\
        .getOrCreate()

    # Load data
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    
    # Read the train and test CSV files
    train_data = spark.read.format("csv").options(header='false', inferschema='true', delimiter=',').load(train_data_path, header=False)
    test_data = spark.read.format("csv").options(header='false', inferschema='true', delimiter=',').load(test_data_path, header=False)
    
    print("Number of rows in the training dataset:", train_data.count())
    print("Number of rows in the test dataset:", test_data.count())
    def rename_columns(df):
        return df.withColumnRenamed("_c0", "age") \
            .withColumnRenamed("_c1", "workclass") \
            .withColumnRenamed("_c2", "fnlwgt") \
            .withColumnRenamed("_c3", "education") \
            .withColumnRenamed("_c4", "education_num") \
            .withColumnRenamed("_c5", "marital_status") \
            .withColumnRenamed("_c6", "occupation") \
            .withColumnRenamed("_c7", "relationship") \
            .withColumnRenamed("_c8", "race") \
            .withColumnRenamed("_c9", "sex") \
            .withColumnRenamed("_c10", "capital_gain") \
            .withColumnRenamed("_c11", "capital_loss") \
            .withColumnRenamed("_c12", "hours_per_week") \
            .withColumnRenamed("_c13", "native_country") \
            .withColumnRenamed("_c14", "income")

    def process_tr_income_column(df):
        return df.withColumn("income", when(col("income") == " >50K", 1.0).otherwise(0.0))
    def process_te_income_column(df):
        return df.withColumn("income", when(col("income") == " >50K.", 1.0).otherwise(0.0))
    # Apply the functions to train and test datasets
    train_data = rename_columns(train_data)
    test_data = rename_columns(test_data)

    train_data = process_tr_income_column(train_data)
    test_data = process_te_income_column(test_data)
    
        # Balance the class distribution in the training data
   # label_col = "income"
    #train_data.filter(col(label_col)).show
   # frac_negative = train_data.filter(col(label_col) == 1).count() / train_data.filter(col(label_col) == 0).count()

  #  train_data = train_data.sampleBy(label_col, fractions={0: 1.0, 1: frac_negative}, seed=42)

        # Balance the class distribution in the training data
    #label_col = "income"
    #train_data.filter(col(label_col)).show
   # frac_negative = test_data.filter(col(label_col) == 1).count() / test_data.filter(col(label_col) == 0).count()

  #  test_data = test_data.sampleBy(label_col, fractions={0: 1.0, 1: frac_negative}, seed=42)

    # Data preprocessing
    # Combine train and test data for preprocessing
    #test_data.show()
    #train_data.show()
    combined_data = train_data.union(test_data)
    # List of categorical and numerical columns
    combined_data.show()
    categorical_columns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
    numerical_columns = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    
    # Index and encode categorical columns
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index", handleInvalid="skip") for column in categorical_columns]
    encoders = [OneHotEncoder(inputCol=column + "_index", outputCol=column + "_encoded") for column in categorical_columns]

    # Assemble the feature vector
    assembler_input = [column + "_encoded" for column in categorical_columns] + numerical_columns
    assembler = VectorAssembler(inputCols=assembler_input, outputCol="features")

    # Scale the feature vector
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
    #income_indexer = StringIndexer(inputCol="income", outputCol="income_index", handleInvalid="skip")
    # Logistic Regression model
    logreg = LogisticRegression(featuresCol="scaled_features", labelCol="income")

    # Pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, logreg])
    # Fit the pipeline
    model = pipeline.fit(train_data)

    # Make predictions
    predictions = model.transform(test_data)
    predictions[['prediction']].show()
    test_data.show()
    train_data.show()
    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="income", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print("Area under ROC curve: ", auc)

    spark.stop()
