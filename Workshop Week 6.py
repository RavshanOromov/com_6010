#!/usr/bin/env python
# coding: utf-8

# ## Workshop Week 6

# ## Logistic Regression
# Breast Cancer data from [the UCI repository](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) contains records corresponding to 
# cases of observed tumors.   There are a number of observations for each and a categorisation in the `class` column: 2 for benign (good), 4 for malignant (bad).  Your task is to build a logistic regression model to classify these cases. 
# 
# The data is provided as a CSV file.  There are a small number of cases where no value is available, these are indicated in the data with `?`. I have used the `na_values` keyword for `read_csv` to have these interpreted as `NaN` (Not a Number).  Your first task is to decide what to do with these rows. You could just drop these rows or you could [impute them from the other data](http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values).
# 
# You then need to follow the procedure outlined in the lecture for generating a train/test set, building and evaluating a model. Your goal is to build the best model possible over this data.   Your first step should be to build a logistic regression model using all of the features that are available.
#   

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE


# In[11]:


# Examine the data: check number of rows and number of columns
import pandas as pd

# Load the data
bcancer = pd.read_csv('/Users/sdfd/Downloads/Practical Week 6 files-20240405 (1)/breast-cancer-wisconsin.csv', na_values='?')
# bcancer = pd.read_csv('Users/sdfd/Downloads/Practical Week 6 files-20240405 (1)/breast_cancer_data.csv', na_values='?')
# Check the first few rows of the dataframe
print(bcancer.head())

# Check the number of rows and columns
print("Number of rows:", bcancer.shape[0])
print("Number of columns:", bcancer.shape[1])


# In[15]:


# # Look at the statistical summary of the dataframe
# import pandas as pd

# # Load the data
# bcancer = pd.read_csv('breast-cancer-wisconsin.csv', na_values='?')

# Check the statistical summary
print(bcancer.describe())

# Check the number of rows and columns
print("Number of rows:", bcancer.shape[0])
print("Number of columns:", bcancer.shape[1])


# In[16]:


# Check how many classes we do have from the "class" column
# Count the number of classes in the "class" column
class_counts = bcancer['class'].value_counts()

print("Number of classes:")
print(class_counts)


# In[17]:


# Check number of samples for each class and comment whether dataset is balanced?
# Count the number of samples for each class
class_counts = bcancer['class'].value_counts()

print("Number of samples for each class:")
print(class_counts)

# Check if the dataset is balanced
is_balanced = abs(class_counts[0] - class_counts[1]) <= 0.1 * bcancer.shape[0]

if is_balanced:
    print("The dataset is balanced.")
else:
    print("The dataset is not balanced.")


# In[18]:


# Deal with the NaN values in the data
# Drop rows with NaN values
bcancer_cleaned = bcancer.dropna()

# Check if any NaN values are left
print("Number of NaN values after dropping:", bcancer_cleaned.isna().sum().sum())

# Check the shape of the cleaned dataframe
print("Shape of the cleaned dataframe:", bcancer_cleaned.shape)


# In[19]:


# Split your data into training(80%) and testing data (20%) and use random_state=142
from sklearn.model_selection import train_test_split

# Separate features (X) and target variable (y)
X = bcancer.drop(columns=['class'])
y = bcancer['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

# Check the shapes of the train and test sets
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)


# In[22]:


from sklearn.impute import SimpleImputer

# Instantiate the imputer with mean strategy
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both training and testing data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Now, you can proceed to train your logistic regression model using the imputed data
logistic_model = LogisticRegression()
logistic_model.fit(X_train_imputed, y_train)

# Make predictions on the imputed testing data
y_pred = logistic_model.predict(X_test_imputed)

# Make predictions on the imputed testing data
y_pred = logistic_model.predict(X_test_imputed)

# Display the predictions
print("Predictions on the test set:")
print(y_pred)


# ### Evaluation
# 
# To evaluate a classification model we want to look at how many cases were correctly classified and how many
# were in error.  In this case we have two outcomes - benign and malignant.   SKlearn has some useful tools, the 
# [accuracy_score]() function gives a score from 0-1 for the proportion correct.  The 
# [confusion_matrix](http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) function 
# shows how many were classified correctly and what errors were made.  Use these to summarise the performance of 
# your model (these functions have already been imported above).

# In[23]:


# Evaluate the performance of your trained model
from sklearn.metrics import accuracy_score, confusion_matrix

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)



# **This is the checkpoint mark for this week's workshop. You need to report `Accuracy Score` on test set and also show `confusion matrix`. You also need to provide analysis based on the results you got.**

# ### Feature Selection
# 
# Since you have many features available, one part of building the best model will be to select which features to use as input to the classifier. Your initial model used all of the features but it is possible that a better model can 
# be built by leaving some of them out.   Test this by building a few models with subsets of the features - how do your models perform? 
# 
# This process can be automated.  The [sklearn RFE function](http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination) implements __Recursive Feature Estimation__ which removes 
# features one by one, evaluating the model each time and selecting the best model for a target number of features.  Use RFE to select features for a model with 3, 4 and 5 features - can you build a model that is as good or better than your initial model?

# In[25]:


from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

# Define logistic regression model
logistic_model = LogisticRegression()

# List of target numbers of features
num_features_list = [3, 4, 5]

# Iterate over target numbers of features
for num_features in num_features_list:
    # Initialize RFE with logistic regression model and number of features
    rfe = RFE(estimator=logistic_model, n_features_to_select=num_features)

    # Fit RFE to the training data
    rfe.fit(X_train_imputed, y_train)

    # Get selected features
    selected_features = X_train.columns[rfe.support_]

    # Print selected features
    print(f"Selected {num_features} features:")
    print(selected_features)

    # Get cross-validated accuracy score
    cv_score = cross_val_score(rfe.estimator_, X_train_imputed[:, rfe.support_], y_train, cv=5)

    # Print cross-validated accuracy score
    print(f"Cross-validated Accuracy (mean): {cv_score.mean():.4f}\n")



# ## Conclusion
# 
# Write a brief conclusion to your experiment.  You might comment on the proportion of __false positive__ and __false negative__ classifications your model makes.  How useful would this model be in a clinical diagnostic setting? 

# In[ ]:


In summary, i built a model to classify breast cancer cases as benign or malignant. 
Initially, this model was quite accurate using all available features. 
However, even with fewer features, it still performed well.

