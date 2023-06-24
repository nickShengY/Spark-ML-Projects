from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
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

    # Data preprocessing
    # Combine train and test data for preprocessing
    combined_data = train_data.union(test_data)
    combined_data.show()
    # List of categorical and numerical columns
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
    
    # Models
    logreg = LogisticRegression(featuresCol="scaled_features", labelCol="income")
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="income")
    dt = DecisionTreeClassifier(featuresCol="scaled_features", labelCol="income")

    # Pipelines
    pipeline_logreg = Pipeline(stages=indexers + encoders + [assembler, scaler, logreg])
    pipeline_rf = Pipeline(stages=indexers + encoders + [assembler, scaler, rf])
    pipeline_dt = Pipeline(stages=indexers +  encoders + [assembler, scaler, dt])

    # Fit the pipelines
    model_logreg = pipeline_logreg.fit(train_data)
    model_rf = pipeline_rf.fit(train_data)
    model_dt = pipeline_dt.fit(train_data)

    # Make predictions
    predictions_logreg = model_logreg.transform(test_data)
    predictions_rf = model_rf.transform(test_data)
    predictions_dt = model_dt.transform(test_data)

    # Evaluate the models
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="income", metricName="areaUnderROC")
    auc_logreg = evaluator
    auc_log = evaluator.evaluate(predictions_logreg)
    auc_rf = evaluator.evaluate(predictions_rf)
    auc_dt = evaluator.evaluate(predictions_dt)
    print("Area under ROC curve for Random Forest Model: ", auc_rf)
    print("Area under ROC curve for Decision Tree Model: ", auc_dt)
    print("For Comparisons: AUC for LogRegression: ", auc_log)
    

    spark.stop()
