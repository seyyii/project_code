#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('project_data.csv')


# In[2]:


df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.nunique()


# In[6]:


# Dropping the 'id' column
df.drop(columns=['id'], inplace=True)


# In[7]:


lowest_age = df['Age'].min()

# Find the highest age
highest_age = df['Age'].max()

print("Lowest Age:", lowest_age)
print("Highest Age:", highest_age)


# In[8]:


# First, we'll ensure 'Age' is in a numeric format for processing
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Define age groups based on common life stages
def categorize_age_group(age):
    if 20 <= age <= 25:
        return 'Young Adult'
    elif 25 < age <= 55:
        return 'Adult'
    elif age > 55:
        return 'Senior'
    else:
        return 'Unknown'  # For any cases where age is missing or not processed correctly

# Apply the age group categorization
df['AgeGroup'] = df['Age'].apply(categorize_age_group)

# Verify the distribution of the new 'AgeGroup' feature
df['AgeGroup'].value_counts()


# In[9]:


# Check for missing values
missing_values = df.isnull().sum()
missing_values


# In[10]:


# checking for outliers
outliers = df
sns.set(style="whitegrid")
outliers.boxplot(figsize=(15,14))


# In[11]:


# Create the boxplot
plt.boxplot(df['Annual_Premium'])

# Set title and labels
plt.title('Boxplot of Annual_Premiun')
plt.xlabel('Annual_Premiun')
plt.ylabel('Values')

# Show the plot
plt.show()


# In[12]:


lowest_Annual_Premium = df['Annual_Premium'].min()

# Find the highest age
highest_Annual_Premium = df['Annual_Premium'].max()

print("Lowest Annual_Premium:", lowest_Annual_Premium)
print("Highest Annual_Premium:", highest_Annual_Premium)


# In[13]:


# Define the percentile threshold for Winsorization
winsor_percentile = 0.95  # You can adjust this based on your preference

# Compute the upper threshold value
upper_threshold = df['Annual_Premium'].quantile(winsor_percentile)

# Winsorize the values above the upper threshold
df['Annual_Premium_winsorized'] = np.where(df['Annual_Premium'] > upper_threshold, upper_threshold, df['Annual_Premium'])

# Now 'Annual_Premium_winsorized' column contains the Winsorized values
# You can use this column for analysis instead of the original 'Annual_Premium'

# Optionally, you can replace the original 'Annual_Premium' column with the winsorized values
df['Annual_Premium'] = df['Annual_Premium_winsorized']

# Drop the intermediate column if needed
df.drop(columns=['Annual_Premium_winsorized'], inplace=True)


# In[14]:


# Create the boxplot
plt.boxplot(df['Annual_Premium'])

# Set title and labels
plt.title('Boxplot of Annual_Premiun')
plt.xlabel('Annual_Premiun')
plt.ylabel('Values')

# Show the plot
plt.show()


# In[15]:


unique_values_count = df['Response'].nunique()

print("Number of unique values in 'response' variable:", unique_values_count)


# In[16]:


unique_values_count = df['Driving_License'].nunique()

print("Number of unique values in 'Driving_License' variable:", unique_values_count)


# In[17]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for column in df.columns:
    if df[column].dtype == 'object':  # Check if column is categorical
        df[column] = label_encoder.fit_transform(df[column])


# In[18]:


df.head()


# In[19]:


df.describe()


# In[20]:


X = df.drop(['Response'], axis=1)
y = df['Response']


# In[21]:


value_counts = df['Response'].value_counts()

# Create a bar chart
plt.figure(figsize=(8, 6))  # Set the figure size (optional)
plt.bar(['0', '1'], [value_counts[0], value_counts[1]], color=['blue', 'green'])


# Set title and labels
plt.title('Bar Chart of Response')
plt.xlabel('Response')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[22]:


from sklearn.model_selection import train_test_split

# First, split into train+validation and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[23]:


# Balancing the response variable
from imblearn.over_sampling import RandomOverSampler

# Instantiate the RandomOverSampler
oversampler = RandomOverSampler(random_state=42)

# Resample the data
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)


# In[24]:


y_resampled.value_counts()


# In[25]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
scaled_features = scaler.fit_transform(X_resampled)

# Apply the same transformation to the validation and test sets
# scaled_vfeatures = scaler.transform(X_validation)
scaled_tfeatures = scaler.transform(X_test)

# Convert scaled_features back to DataFrame
X_train_scaled = pd.DataFrame(scaled_features, columns=X.columns)
# X_validation_scaled = pd.DataFrame(scaled_vfeatures, columns=X.columns)
X_test_scaled = pd.DataFrame(scaled_tfeatures, columns=X.columns)

scaled_df = pd.concat([pd.DataFrame(y_resampled), pd.DataFrame(scaled_features)], axis=1)


# In[26]:


scaled_df.head()


# In[27]:


from sklearn.linear_model import LogisticRegression

# L1 regularization (Lasso) feature selection
lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
lasso.fit(X_train_scaled, y_resampled)
selected_features_lasso = X.columns[abs(lasso.coef_[0]) > 0]
selected_features_lasso


# In[28]:


from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import itertools


# In[29]:


# Create the classifiers
clf_rf = RandomForestClassifier()
clf_dt = DecisionTreeClassifier()
clf_lr = LogisticRegression()

# Cross-validation
cv_scores_rf = cross_val_score(clf_rf, X_train_scaled, y_resampled, cv=5, scoring='accuracy')
cv_scores_dt = cross_val_score(clf_dt, X_train_scaled, y_resampled, cv=5, scoring='accuracy')
cv_scores_lr = cross_val_score(clf_lr, X_train_scaled, y_resampled, cv=5, scoring='accuracy')

# Fit and predict with Random Forest Classifier
clf_rf.fit(X_train_scaled, y_resampled)
y_pred_rf = clf_rf.predict(X_test_scaled)

# Fit and predict with Decision Tree Classifier
clf_dt.fit(X_train_scaled, y_resampled)
y_pred_dt = clf_dt.predict(X_test_scaled)

# Fit and predict with Logistic Regression
clf_lr.fit(X_train_scaled, y_resampled)
y_pred_lr = clf_lr.predict(X_test_scaled)

# Calculate evaluation metrics for Random Forest Classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)

# Calculate evaluation metrics for Decision Tree Classifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)

# Calculate evaluation metrics for Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_pred_lr)

# Print the evaluation metrics
print("Random Forest Classifier Metrics:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1-score:", f1_rf)
print("ROC-AUC score:", roc_auc_rf)
print("Cross-Validation Accuracy (Random Forest):", np.mean(cv_scores_rf))

print("\nDecision Tree Classifier Metrics:")
print("Accuracy:", accuracy_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1-score:", f1_dt)
print("ROC-AUC score:", roc_auc_dt)
print("Cross-Validation Accuracy (Decision Tree):", np.mean(cv_scores_dt))

print("\nLogistic Regression Metrics:")
print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1-score:", f1_lr)
print("ROC-AUC score:", roc_auc_lr)
print("Cross-Validation Accuracy (Logistic Regression):", np.mean(cv_scores_lr))

# Classification report for Random Forest Classifier
print("\nClassification Report (Random Forest Classifier):")
print(classification_report(y_test, y_pred_rf))

# Classification report for Decision Tree Classifier
print("\nClassification Report (Decision Tree Classifier):")
print(classification_report(y_test, y_pred_dt))

# Classification report for Logistic Regression
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

# Confusion matrix for Random Forest Classifier
print("\nConfusion Matrix (Random Forest Classifier):")
print(confusion_matrix(y_test, y_pred_rf))

# Confusion matrix for Decision Tree Classifier
print("\nConfusion Matrix (Decision Tree Classifier):")
print(confusion_matrix(y_test, y_pred_dt))

# Confusion matrix for Logistic Regression
print("\nConfusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_pred_lr))

# Plot confusion matrix for Random Forest Classifier
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Interested', 'Interested'], yticklabels=['Not Interested', 'Interested'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.show()

# Plot confusion matrix for Decision Tree Classifier
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Interested', 'Interested'], yticklabels=['Not Interested', 'Interested'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.show()

# Plot confusion matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Interested', 'Interested'], yticklabels=['Not Interested', 'Interested'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()


# Random Forest Classifier Metrics:
# 
# Accuracy: 84.65% of the predictions made by the Random Forest model are correct.
# Precision: 35.33% of the transactions predicted as interested by the model are actually interested.
# Recall: 27.54% of the actual interested transactions were correctly identified by the model.
# F1-score: 30.96% is the harmonic mean of precision and recall, which balances the two metrics.
# ROC-AUC score: 60.17% is the area under the Receiver Operating Characteristic curve, which measures the model's ability to distinguish between classes.
# Cross-Validation Accuracy (Random Forest): 95.03% is the average accuracy of the model across different cross-validation folds, indicating the overall performance of the model on unseen data.
# 
# 
# Decision Tree Classifier Metrics:
# 
# Accuracy: 82.80% of the predictions made by the Decision Tree model are correct.
# Precision: 30.04% of the transactions predicted as interested by the model are actually interested.
# Recall: 28.34% of the actual interested transactions were correctly identified by the model.
# F1-score: 29.17% is the harmonic mean of precision and recall, which balances the two metrics.
# ROC-AUC score: 59.46% is the area under the Receiver Operating Characteristic curve, which measures the model's ability to distinguish between classes.
# Cross-Validation Accuracy (Decision Tree): 94.39% is the average accuracy of the model across different cross-validation folds, indicating the overall performance of the model on unseen data.
# 
# 
# Logistic Regression Metrics:
# 
# Accuracy: 67.90% of the predictions made by the Logistic Regression model are correct.
# Precision: 27.29% of the transactions predicted as interested by the model are actually interested.
# Recall: 94.33% of the actual interested transactions were correctly identified by the model.
# F1-score: 42.34% is the harmonic mean of precision and recall, which balances the two metrics.
# ROC-AUC score: 79.23% is the area under the Receiver Operating Characteristic curve, which measures the model's ability to distinguish between classes.
# Cross-Validation Accuracy (Logistic Regression): 78.96% is the average accuracy of the model across different cross-validation folds, indicating the overall performance of the model on unseen data.
# 
# 
# Based on these metrics, the choice of which model to use depends on the specific requirements of your application. If you prioritize accuracy and generalization to unseen data, the Random Forest model may be the best choice due to its highest accuracy and cross-validation accuracy. However, if you prioritize interpretability and simplicity, the Decision Tree model may be preferred. On the other hand, if you prioritize the ability to identify interested transactions accurately, even at the cost of some misclassifications, the Logistic Regression model with its high recall may be the best choice.
# 
# 
# 
# 
# 

# Random Forest Classifier Confusion Matrix:
# 
# lua
# Copy code
# [[61898  4801]
#  [ 6900  2623]]
# True Negatives (TN): 61,898
# False Positives (FP): 4,801
# False Negatives (FN): 6,900
# True Positives (TP): 2,623
# Interpretation:
# 
# The Random Forest model correctly classified 61,898 transactions as "Not Interested" (TN).
# It incorrectly classified 4,801 transactions as "Interested" when they were actually "Not Interested" (FP).
# It incorrectly classified 6,900 transactions as "Not Interested" when they were actually "Interested" (FN).
# It correctly classified 2,623 transactions as "Interested" (TP).
# Decision Tree Classifier Confusion Matrix:
# 
# lua
# Copy code
# [[60414  6285]
#  [ 6824  2699]]
# True Negatives (TN): 60,414
# False Positives (FP): 6,285
# False Negatives (FN): 6,824
# True Positives (TP): 2,699
# Interpretation:
# 
# The Decision Tree model correctly classified 60,414 transactions as "Not Interested" (TN).
# It incorrectly classified 6,285 transactions as "Interested" when they were actually "Not Interested" (FP).
# It incorrectly classified 6,824 transactions as "Not Interested" when they were actually "Interested" (FN).
# It correctly classified 2,699 transactions as "Interested" (TP).
# Logistic Regression Confusion Matrix:
# 
# lua
# Copy code
# [[42768 23931]
#  [  540  8983]]
# True Negatives (TN): 42,768
# False Positives (FP): 23,931
# False Negatives (FN): 540
# True Positives (TP): 8,983
# Interpretation:
# 
# The Logistic Regression model correctly classified 42,768 transactions as "Not Interested" (TN).
# It incorrectly classified 23,931 transactions as "Interested" when they were actually "Not Interested" (FP).
# It incorrectly classified 540 transactions as "Not Interested" when they were actually "Interested" (FN).
# It correctly classified 8,983 transactions as "Interested" (TP).
# 
# 
# In summary, the confusion matrices provide insights into how each model is performing in terms of correctly and incorrectly classifying transactions. We can use these matrices to understand the strengths and weaknesses of each classifier and make an informed decision about which model to use based on our specific requirements and priorities.
# 
# 
# 
# 
# 

# 
# Random Forest Classifier:
# 
# Cross-Validation Accuracy: 0.9503
# This model achieves the highest cross-validation accuracy among the three. Cross-validation accuracy is a measure of how well the model generalizes to unseen data. A higher cross-validation accuracy indicates better generalization performance. Therefore, Random Forest Classifier is likely to perform well on new, unseen data.
# 
# 
# Decision Tree Classifier:
# 
# Cross-Validation Accuracy: 0.9439
# The Decision Tree Classifier has a slightly lower cross-validation accuracy compared to the Random Forest Classifier. While still performing well, it may not generalize as effectively as the Random Forest Classifier to unseen data.
# 
# 
# Logistic Regression:
# 
# Cross-Validation Accuracy: 0.7896
# Logistic Regression has the lowest cross-validation accuracy among the three models. This suggests that it may not generalize as well to new data compared to Random Forest and Decision Tree Classifiers. However, it's essential to note that logistic regression may still be suitable depending on the specific requirements of the problem and the trade-offs between accuracy and interpretability.
# In summary, based on cross-validation accuracy alone, the Random Forest Classifier appears to be the best-performing model as it achieves the highest accuracy on unseen data.

# In[30]:


from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[31]:


# Create the Gradient Boosting classifier
clf_gb = GradientBoostingClassifier()

# Cross-validation
cv_scores_gb = cross_val_score(clf_gb, X_train_scaled, y_resampled, cv=5, scoring='accuracy')

# Fit and predict with Gradient Boosting
clf_gb.fit(X_train_scaled, y_resampled)
y_pred_gb = clf_gb.predict(X_test_scaled)

# Calculate evaluation metrics for Gradient Boosting
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
roc_auc_gb = roc_auc_score(y_test, y_pred_gb)

# Print the evaluation metrics
print("Gradient Boosting Classifier Metrics:")
print("Accuracy:", accuracy_gb)
print("Precision:", precision_gb)
print("Recall:", recall_gb)
print("F1-score:", f1_gb)
print("ROC-AUC score:", roc_auc_gb)
print("Cross-Validation Accuracy (Gradient Boosting):", np.mean(cv_scores_gb))

# Classification report for Gradient Boosting
print("\nClassification Report (Gradient Boosting):")
print(classification_report(y_test, y_pred_gb))

# Confusion matrix for Gradient Boosting
print("\nConfusion Matrix (Gradient Boosting):")
print(confusion_matrix(y_test, y_pred_gb))

# Plot confusion matrix for Gradient Boosting
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_gb), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Interested', 'Interested'], yticklabels=['Not Interested', 'Interested'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Gradient Boosting')
plt.show()


# In[32]:


# Create the K-Nearest Neighbors Classifier
clf_knn = KNeighborsClassifier()

# Cross-validation
cv_scores_knn = cross_val_score(clf_knn, X_train_scaled, y_resampled, cv=5, scoring='accuracy')

# Fit and predict with K-Nearest Neighbors Classifier
clf_knn.fit(X_train_scaled, y_resampled)
y_pred_knn = clf_knn.predict(X_test_scaled)

# Calculate evaluation metrics for K-Nearest Neighbors Classifier
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
roc_auc_knn = roc_auc_score(y_test, y_pred_knn)

# Print the evaluation metrics
print("K-Nearest Neighbors Classifier Metrics:")
print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1-score:", f1_knn)
print("ROC-AUC score:", roc_auc_knn)
print("Cross-Validation Accuracy (K-Nearest Neighbors Classifier):", np.mean(cv_scores_knn))

# Classification report for K-Nearest Neighbors Classifier
print("\nClassification Report (K-Nearest Neighbors Classifier):")
print(classification_report(y_test, y_pred_knn))

# Confusion matrix for K-Nearest Neighbors Classifier
print("\nConfusion Matrix (K-Nearest Neighbors Classifier):")
print(confusion_matrix(y_test, y_pred_knn))

# Plot confusion matrix for K-Nearest Neighbors Classifier
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Interested', 'Interested'], yticklabels=['Not Interested', 'Interested'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - K-Nearest Neighbors Classifier')
plt.show()


# In[2]:


import matplotlib.pyplot as plt

# Classifier names
classifiers = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting', 'K-Nearest Neighbors']

# Accuracy scores
accuracy_scores = [0.8453727270341896, 0.8280286531447614, 0.6789509590406969, 0.7048621133006219, 0.758849151163706]

# Create bar chart for accuracy scores
plt.figure(figsize=(10, 6))
plt.bar(classifiers, accuracy_scores, color='skyblue')
plt.title('Accuracy Scores of The Different Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy scores
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[3]:


import matplotlib.pyplot as plt

# Classifier names and precision scores
classifiers = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting', 'KNN']
precision_scores = [0.3484259879437374, 0.3005008347245409, 0.27292337607097283, 0.2880852036982587, 0.28514601726981664]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(classifiers, precision_scores, color='skyblue')

# Add labels and title
plt.xlabel('Classifier')
plt.ylabel('Precision Score')
plt.title('Precision Scores for The Different Classifiers')

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[4]:


import matplotlib.pyplot as plt

# Classifier names and F1 scores
classifiers = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting', 'KNN']
f1_scores = [0.30621615257829055, 0.2917657229306246, 0.4233569762235785, 0.439449815608492, 0.3900852772339649]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(classifiers, f1_scores, color='lightgreen')

# Add labels and title
plt.xlabel('Classifier')
plt.ylabel('F1 Score')
plt.title('F1 Scores for The Different Classifiers')

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[5]:


import matplotlib.pyplot as plt

# Classifier names and recall scores
classifiers = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting', 'KNN']
recall_scores = [0.273128215898, 0.2917657229306246, 0.9432951800903077, 0.9259687073401239, 0.6172424656095769]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(classifiers, recall_scores, color='lightblue')

# Add labels and title
plt.xlabel('Classifier')
plt.ylabel('Recall Score')
plt.title('Recall Scores for The Different Classifiers')

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[6]:


import matplotlib.pyplot as plt

# Classifier names and cross-validation accuracy scores
classifiers = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting', 'KNN']
cv_accuracy_scores = [0.9502820321255137, 0.9439166977960405, 0.789641389615241, 0.7978427344041839, 0.8687579379902877]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(classifiers, cv_accuracy_scores, color='lightgreen')

# Add labels and title
plt.xlabel('Classifier')
plt.ylabel('Cross-Validation Accuracy Score')
plt.title('Cross-Validation Accuracy Scores for The Different Classifiers')

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:




