#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES AND DATASET

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
import string
import re
import tensorflow as tf
from surprise import Dataset, Reader
from sklearn.model_selection import train_test_split
from surprise import KNNBasic
from surprise.accuracy import rmse
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
import researchpy as rp
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import squarify
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# Read the Excel file into a DataFrame
data = pd.read_excel("C:/Users/Administrator/Desktop/Copy of Job Recommendation System (1)(3).xlsx")

# display the dataframe
data.head()


# In[2]:


data.shape


# In[3]:


data.info()


# # DESCRIPTIVE STATISTICS

# In[4]:


# Get descriptive statistics
descriptive_stats = data.describe()

# Display the descriptive statistics
print(descriptive_stats)


# # CHECKING FOR MISSING VALUES

# In[5]:


# Check for missing values in the dataset
missing_values = data.isnull().sum()

# Display the count of missing values for each column
print(missing_values)


# In[6]:


# Impute missing values for numerical columns with the mean
numerical_columns = data.select_dtypes(include='number').columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# Impute missing values for categorical columns with the most frequent value
categorical_columns = data.select_dtypes(include='object').columns
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])


# In[7]:


# Check for missing values in the dataset
missing_values = data.isnull().sum()

# Display the count of missing values for each column
print(missing_values)


# In[8]:


# Plot histogram of Age
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], bins=20, kde=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram of Age')
plt.show()


# In[9]:


# Plot count of Gender
plt.figure(figsize=(6, 4))
sns.countplot(data['Gender'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Count of Gender')
plt.show()


# In[10]:


# Plot count of Academic Background
plt.figure(figsize=(10, 6))
sns.countplot(data['Academic Background'])
plt.xlabel('Academic Background')
plt.ylabel('Count')
plt.title('Count of Academic Background')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[11]:


# Plot count of Job Type Interest
plt.figure(figsize=(8, 5))
sns.countplot(data['Job Type Interest'])
plt.xlabel('Job Type Interest')
plt.ylabel('Count')
plt.title('Count of Job Type Interest')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[12]:


# Plot box plot of Salary Expectation
plt.figure(figsize=(8, 6))
sns.boxplot(data['Salary Expectation (in USD)'])
plt.xlabel('Salary Expectation (in USD)')
plt.title('Box Plot of Salary Expectation')
plt.show()


# In[13]:


# Plot correlation heatmap for numerical columns
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[14]:


plt.figure(figsize=(10, 6))
sns.countplot(data['Industry Interest'])
plt.xlabel('Industry Interest')
plt.ylabel('Count')
plt.title('Count of Students in Each Industry of Interest')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[15]:


sns.pairplot(data, vars=['Age', 'Salary Expectation (in USD)'], hue='Gender')
plt.suptitle('Pairplot of Age and Salary Expectation (colored by Gender)')
plt.show()


# In[16]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='Job Type Interest', y='Salary Expectation (in USD)', data=data)
plt.xlabel('Job Type Interest')
plt.ylabel('Salary Expectation (in USD)')
plt.title('Distribution of Salary Expectation across Job Types')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[17]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Academic Background', y='Salary Expectation (in USD)', data=data)
plt.xlabel('Academic Background')
plt.ylabel('Salary Expectation (in USD)')
plt.title('Comparison of Salary Expectation for Different Academic Backgrounds')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[18]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Salary Expectation (in USD)', data=data, hue='Gender')
plt.xlabel('Age')
plt.ylabel('Salary Expectation (in USD)')
plt.title('Scatter Plot of Age vs. Salary Expectation (colored by Gender)')
plt.show()


# In[19]:


# Method 1: One-Hot Encoding
one_hot_encoded_data = pd.get_dummies(data, columns=['Academic Background', 'Field of Study', 'Industry Interest', 'Job Type Interest', 'Location Interest', 'desired company'], drop_first=True)


label_encoder = LabelEncoder()
data['Gender_encoded'] = label_encoder.fit_transform(data['Gender'])

# Display the encoded datasets
print("One-Hot Encoded Data:")
print(one_hot_encoded_data.head())

print("\nLabel Encoded Data:")
print(data[['Gender', 'Gender_encoded']].head())


# In[20]:


# Feature 1: Extracting First Name from Name
data['First Name'] = data['Name'].str.split().str[0]

# Feature 2: Converting 'Age' to Age Group
def get_age_group(age):
    if age < 25:
        return 'Young'
    elif age >= 25 and age < 40:
        return 'Middle-aged'
    else:
        return 'Senior'

data['Age Group'] = data['Age'].apply(get_age_group)

# Feature 3: Counting the number of skills each student has
data['Number of Skills'] = data['Skills'].apply(lambda x: len(x.split(',')))

# Feature 4: Encoding 'Full-time' as 1 and 'Part-time' as 0 for Job Type Interest
data['Job Type Interest'] = data['Job Type Interest'].apply(lambda x: 1 if x == 'Full-time' else 0)

# Feature 5: Encoding 'Male' as 1 and 'Female' as 0 for Gender
data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Display the updated dataset with engineered features
print(data.head())


# In[21]:


# Prepare the feature matrix X and target vector y
X = data[['Age', 'Gender']]  # Select relevant features
y = data['Job Type Interest']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print("KNN Classifier Accuracy:", accuracy)


# In[22]:


import tensorflow as tf
# Prepare the feature matrix X and target vector y
X = data[['Age', 'Gender', 'Number of Skills']]  # Select relevant features
y = data['Job Type Interest']  # Target variable

# One-hot encode the target variable for binary classification
y = pd.get_dummies(y, drop_first=True)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data for better training performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_class = np.round(y_pred)  # Convert probabilities to binary classes

# Convert predictions back to original class labels for evaluation
y_pred_labels = pd.DataFrame(y_pred_class, columns=y.columns, index=y_test.index)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred_labels)
print("Neural Network Accuracy:", accuracy)


# In[23]:


# Import required libraries
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise.accuracy import rmse

# Prepare the data for Surprise library
reader = Reader(rating_scale=(0, 1))
surprise_data = Dataset.load_from_df(data[['user ID', 'Salary Expectation (in USD)', 'Age']], reader)

# Split the data into training and testing sets (80% train, 20% test)
trainset, testset = train_test_split(surprise_data, test_size=0.2, random_state=42)

# Initialize and train the KNNBasic collaborative filtering model
knn_collaborative = KNNBasic(sim_options={'user_based': True})
knn_collaborative.fit(trainset)

# Make predictions on the test set
predictions = knn_collaborative.test(testset)

# Evaluate the model's performance (Root Mean Squared Error, RMSE)
rmse_score = rmse(predictions)
print("Collaborative Filtering RMSE:", rmse_score)


# In[24]:


# Import required libraries
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise.accuracy import rmse

# Prepare the data for Surprise library
reader = Reader(rating_scale=(0, 1))
surprise_data = Dataset.load_from_df(data[['user ID', 'Salary Expectation (in USD)', 'Age']], reader)

# Split the data into training and testing sets (80% train, 20% test)
trainset, testset = train_test_split(surprise_data, test_size=0.2, random_state=42)

# Initialize and train the KNNBasic collaborative filtering model
knn_collaborative = KNNBasic(sim_options={'user_based': True})
knn_collaborative.fit(trainset)

# Make predictions on the test set
predictions = knn_collaborative.test(testset)

# Evaluate the model's performance (Root Mean Squared Error, RMSE)
rmse_score = rmse(predictions)
print("Collaborative Filtering RMSE:", rmse_score)


# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Prepare the feature matrix X and target vector y
X = data[['Age', 'Gender', 'Number of Skills']]  # Select relevant features
y = data['Job Type Interest']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data for better training performance (not necessary for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models for different algorithms
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Calculate ROC curves and AUC for each model
models = [knn_classifier, rf_classifier]
model_names = ['KNN Classifier', 'Random Forest']

plt.figure(figsize=(10, 8))

for model, name in zip(models, model_names):
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Different Algorithms')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[30]:


accuracy = accuracy_score(y_test, y_pred_prob.round())  # Calculate accuracy using y_pred_prob
report = classification_report(y_test, knn_classifier.predict(X_test_scaled))  # Using the KNN classifier

print(f"--- {name} ---")
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
print("-------------------------------------")


# In[31]:


# Get feature importances from the trained Random Forest classifier
feature_importances = rf_classifier.feature_importances_

# Print feature importances
print("Feature Importances:")
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")


# In[32]:


# Get feature importances from the trained Random Forest classifier
feature_importances = rf_classifier.feature_importances_

# Plot feature importances in a bar plot
feature_names = X.columns
plt.figure(figsize=(8, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importances')
plt.show()


# In[53]:


import joblib
import os

# Create the directory
output_directory = 'student_job_recommendation_system/'
os.makedirs(output_directory, exist_ok=True)

# Convert categorical variables to numerical using one-hot encoding
encoder = OneHotEncoder()
academic_encoded = encoder.fit_transform(academic_data).toarray()

# Scale the data for better clustering performance
scaler = StandardScaler()
academic_scaled = scaler.fit_transform(academic_encoded)

# Save the trained machine learning models
# Assuming you already have the trained models: knn_classifier, knn_collaborative, bayesian_model, model, rf_classifier
joblib.dump(knn_classifier, 'student_job_recommendation_system/knn_classifier_model.pkl')
joblib.dump(rf_classifier, 'student_job_recommendation_system/random_forest_model.pkl')


# In[ ]:


# Perform K-Means clustering
num_clusters = 5  # Change this number based on the desired number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(academic_scaled)

# Save the K-Means clustering model
joblib.dump(kmeans, 'student_job_recommendation_system/kmeans_model.pkl')

