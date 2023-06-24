# Part3-Logistic Regression classifier on Census Income Data
# Introduction

In this part, we aimed at training a binary classification and predicting an individual's income level using the Census dataset from the UCI Machine Learning Repository, based on various demographic and economic features.

# Dataset Description

**This data set must place at dir: "/fp/data/p3d", and the file name must be "train.csv" and "test.csv"!**

The dataset on Census Dataset consists of anonymized information such as age, occupation, education, and working class, with the goal of training a binary classifier to predict income levels, either >50K or <50K. With 48,842 instances and 14 attributes, the dataset offers a diverse mix of categorical, numerical, and missing values. We utilized Spark ML/MLlib's logistic regression capabilities to train our model and evaluate its performance, aiming to achieve a high accuracy rate.

# Model Explanation and Results

What we need to do for this task was logistic regression, which was also implemented using the PySpark 'LogisticRegression' class. The output of the logistic regression model was measured using the area under the receiver operating characteristic curve (ROC AUC). In this case, the code had produced an ROC AUC score of 0.904, which indicated that the Logistic Regression model had good performance in training the binary classification and predicting the income of the individuals.
