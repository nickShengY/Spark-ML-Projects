# P4- Random Forest and Decision Tree Classifier on Census Income Data
# Intrduction

In this part, we would revisit the third part's problem but instead employ Random Forest Classifier and Decision Tree Classifier using Apache Spark ML/MLlib. This approach would enable us to compare the performance of different machine learning algorithms on the same dataset, further enhancing our understanding of their effectiveness and suitability for specific tasks.

# Dataset Description

**This data set must place at dir: "/fp/data/p3d", and the file name must be "train.csv" and "test.csv"!**

The dataset on Census Dataset consists of anonymized information such as age, occupation, education, and working class, with the goal of training a binary classifier to predict income levels, either >50K or <50K. With 48,842 instances and 14 attributes, the dataset offers a diverse mix of categorical, numerical, and missing values. We utilized Spark ML/MLlib's logistic regression capabilities to train our model and evaluate its performance, aiming to achieve a high accuracy rate.

# Model Explanation and Results

We implemented another two different classification models: Random Forest and Decision Tree classifiers. Both of the two models were trained on the preprocessed and standardized features in the training dataset, and predictions were made on the test dataset. The performance of the Random Forest and Decision Tree models was evaluated using the area under the receiver operating characteristic (ROC) curve. We got the AUC of Random Forest was 0.89, and the AUC of Decision Tree was 0.59. Compared with the AUC score of Logistic Regression we got from P3(0.904), we could say Logistic Regression performed best among these three models in predicting the personal income, and the Decision Tree made the worth performance.
