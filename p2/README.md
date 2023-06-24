# Part2-Heart Disease Prediction using Logistic Regressionx
# Introduction

In this part, we designed a Python script used Logistic Regression to identify the most significant risk factors associated with heart disease and predicted overall risk levels. In this question, we would be using the Framingham Heart dataset.

# Dataset Description

**This data set must place at dir: "/fp/data/p2d", and the file name must be "framingham.csv" and "heart.csv"!**

The Framingham Heart Dataset is originated from the Framingham Heart Study, with an initial cohort of 5209 subjects. The Framingham Heart dataset comprises various demographic, clinical, and laboratory attributes collected from the study participants. The data typically includes various demographic information about patients, such as age, sex, BMI, the measurement information of patients such as heart rate, total cholesterol level, systolic blood pressure, diastolic blood pressure, and fasting blood pressure etc., and the outcome of the presence or absence of coronary heart disease. This dataset enables researchers to investigate the relationship between various risk factors. 

# Model Explanation and Results

The chosen model for this task was logistic regression, so we implemented using the PySpark 'LogisticRegression' class. Logistic regression is a popular choice for binary classification tasks as it provides an interpretable and efficient method for predicting probabilities. The model was trained on the preprocessed and standardized features in the training dataset, and predictions were made on the test dataset. The performance of the logistic regression model was evaluated using the area under the receiver operating characteristic (ROC) curve, and we got the AUC score with 0.752, which meant that the model was able to distinguish between positive and negative classes with a moderate performance.
