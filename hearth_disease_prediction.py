# Heart Disease Prediction using Logistic Regression


# Work Flow
# 1. Importing Libraries
# 2. Collecting Data
# 3. Data Preprocessing
# 4. Splitting the Data
# 5. Model Training
# 6. Model Evaluation
# 7. Making Predictions

# Model: Logistic Regression because it is effective for binary classification tasks like predicting the presence or absence of heart disease.

# DataSet Structure:
# age: Age of the patient
# sex: Sex of the patient
# cp: Chest pain type
# trestbps: Resting blood pressure
# chol: Serum cholesterol in mg/dl
# fbs: Fasting blood sugar > 120 mg/dl
# restecg: Resting electrocardiographic results
# thalach: Maximum heart rate achieved
# exang: Exercise induced angina
# oldpeak: ST depression induced by exercise relative to rest
# slope: Slope of the peak exercise ST segment
# ca: Number of major vessels (0-3) colored by fluoroscopy
# thal: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
# target: Diagnosis of heart disease (1 = presence; 0 = absence)

# ==============================================
# 1. Importing Libraries
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================================
# 2. Collecting Data
# ==============================================

data_df = pd.read_csv("heart_disease_data.csv")

# ==============================================
# 3. Data Preprocessing
# ==============================================

# Print the first few rows of the dataset
print(data_df.head())

# Print the shape of the dataset
print(data_df.shape)

# Check for missing values
print(data_df.isnull().sum())

# Print data description
print(data_df.describe())

# Check the distribution of the target variable
print(data_df["target"].value_counts())

# Print Correlation Matrix
print(data_df.corr())

# plt.figure(figsize=(10, 8))
# sns.heatmap(data_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.show()

# ==============================================
# 4. Splitting the Data
# ==============================================

X = data_df.drop(columns="target", axis=1)
y = data_df["target"]

print(X)
print(y)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)

# ==============================================
# 5. Model Training
# ==============================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==============================================
# 6. Model Evaluation
# ==============================================

# Training Data Evaluation
train_data_pred = model.predict(X_train)

train_accuracy = accuracy_score(y_train, train_data_pred)

print("Training Data Accuracy:", train_accuracy)

confusion_matrix_train = confusion_matrix(y_train, train_data_pred)

print("Confusion Matrix (Train):\n", confusion_matrix_train)

classification_report_train = classification_report(y_train, train_data_pred)

print("Classification Report (Train):\n", classification_report_train)

# Conclusion:
# Accuracy: 0.8553719008264463 (85.54%)
# The model performs well on the training data, indicating it has learned the patterns effectively.
# Confusion Matrix:  [[ 85  25][ 10 122]] - This shows the number of correct and incorrect predictions made by the model.
# Classification Report: Precision, Recall, F1-Score values are all above 0.8, indicating good performance across all metrics.

# Test Data Evaluation
test_data_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_data_pred)

print("Test Data Accuracy:", test_accuracy)

confusion_matrix_test = confusion_matrix(y_test, test_data_pred)

print("Confusion Matrix (Test):\n", confusion_matrix_test)

classification_report_test = classification_report(y_test, test_data_pred)

print("Classification Report (Test):\n", classification_report_test)

# Conclusion:
# Accuracy: 0.8032786885245902 (80.33%)
# The model performs well on the test data, indicating good generalization.
# Confusion Matrix: [[22  6][ 6 27]] - This shows the number of correct and incorrect predictions made by the model.
# Classification Report: Precision, Recall, F1-Score values are all above 0.

# Check for overfitting or underfitting

if abs(train_accuracy - test_accuracy) < 0.1:
    print("The model is neither overfitting nor underfitting.")
else:
    print("The model may be overfitting or underfitting.")


# Conclusion:
# The model shows similar performance on both training and test datasets, indicating it is well-balanced without
# significant overfitting or underfitting.
# Overfitting occurs when a model learns the training data too well, capturing noise and details that do not generalize to new data.
# Underfitting occurs when a model is too simple to capture the underlying patterns in the data.
# For example, if the training accuracy was significantly higher than the test accuracy, it would suggest overfitting.
# Conversely, if both accuracies were low, it would suggest underfitting.

# ==============================================
# 7. Making Predictions
# ==============================================

input_data = (57, 1, 2, 128, 229, 0, 0, 150, 0, 0.4, 1, 1, 3)

# Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("The person does not have heart disease.")
else:
    print("The person has heart disease.")
