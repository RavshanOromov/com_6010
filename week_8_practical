#!/usr/bin/env python
# coding: utf-8

# # Portfolio Part 3 - Analysis of Mobile Price Data (2024 S1)

# In this Portfolio task, you will work on a new dataset named 'Mobile Price Data', it contains numerous details about mobile phone hardware, specifications, and prices. Your main task is to train classification models to predict **mobile phone prices** ('price range' in the dataset)and evaluate the strengths and weaknesses of these models.

# Here's the explanation of each column:

# |Column|Meaning|
# |:-----:|:-----:|
# |battery power|Total energy a battery can store in one time measured in mAh|
# |blue|Has bluetooth or not|
# |clock speed|speed at which microprocessor executes instructions|
# |dual sim|Has dual sim support or not|
# |fc|Front Camera mega pixels|
# |four g|Has 4G or not|
# |int memory|Internal Memory in Gigabytes|
# |m dep|Mobile Depth in cm|
# |mobile wt|Weight of mobile phone|
# |n cores|Number of cores of processor|
# |pc|Primary Camera mega pixels|
# |px height|Pixel Resolution Height|
# |px width|Pixel Resolution Width|
# |ram|Random Access Memory in Mega Bytes|
# |sc h|Screen Height of mobile in cm|
# |sc w|Screen Width of mobile in cm|
# |talk time|longest time that a single battery charge will last when you are|
# |three g|Has 3G or not|
# |touch screen|Has touch screen or not|
# |wifi|Has wifi or not|
# |price range|This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost)|

# Blue, dual sim, four g, three g, touch screen, and wifi are all binary attributes, 0 for not and 1 for yes.

# Your high level goal in this notebook is to build and evaluate predictive models for 'price range' from other available features. More specifically, you need to **complete the following major steps**:
# 
# 1. ***Explore the data*** and ***clean the data if necessary***. For example, remove abnormal instanaces and replace missing values.
# 
# 2. ***Study the correlation*** between 'price range' with other features. And ***select the variables*** that you think are helpful for predicting the price range. We do not limit the number of variables.
# 
# 3. ***Split the dataset*** (Trainging set : Test set = 8 : 2)
# 
# 4. ***Train a logistic regression model*** to predict 'price range' based on the selected features (from the second step). ***Calculate the accuracy*** of your model. (You are required to report the accuracy from both training set and test set.) ***Explain your model and evaluate its performance*** (Is the model performing well? If yes, what factors might be contributing to the good performance of your model? If not, how can improvements be made?).
# 
# 5. ***Train a KNN model*** to predict 'price range' based on the selected features (you can use the features selected from the second step and set K with an ad-hoc manner in this step. ***Calculate the accuracy*** of your model. (You are required to report the accuracy from both training set and test set.)
# 
# 6. ***Tune the hyper-parameter K*** in KNN (Hints: GridsearchCV), ***visualize the results***, and ***explain*** how K influences the prediction performance.
# 
#   Hints for visualization: You can use line chart to visualize K and mean accuracy scores on test set.

# Note 1: In this assignment, we no longer provide specific guidance and templates for each sub task. You should learn how to properly comment your notebook by yourself to make your notebook file readable.
# 
# Note 2: You will not being evaluated on the accuracy of the model but on the process that you use to generate it and your explanation.

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# In[27]:


# read the csv file

df = pd.read_csv('/Users/sdfd/Downloads/Mobile_Price_Data.csv')
df.head(10)


# In[28]:


df.info()


# In[29]:


#remove missing dataset
print('sum of null data in each column')
print(df.isnull().sum())

print('Length before removing missing data', len(df))
df.dropna(inplace=True)

print('Sum of null data in each column after cleaning ')
print(df.isnull().sum())
print("Length after removing missing data", len(df))


# In[30]:


#Data exploration after clean dataset
df.describe()


# In[49]:


print(df.corr())
#................
#ram, battery_power, px_width, px-height, int_memory


# In[62]:


#Split the data set

train, test = train_test_split(df, test_size = 0.2, random_state = 142)
print('Train set shape', train.shape)
print('Test set shape', test.shape)


# In[63]:


X_train = train[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory']]
y_train = train[['price_range']]

X_test = test[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory']]
y_test = test[['price_range']]


# In[64]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[66]:


#4 Logistic Regression

lr = LogisticRegression().fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)


# train_accuracy = accuracy_score(Y_train, train_predictions)


# In[67]:


#Calculation accuracy on training set

print('Accursacy of traing set', accuracy_score(y_train, y_pred_train))



#Calculating accuracy on test set
print('Accuracy of test set', accuracy_score(y_test, y_pred_test))


# In[68]:


#KNN


# In[70]:


from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrices import accuracy_score



# In[92]:


#create KNN classifier model

clf = KNeighborsClassifier(n_neighbors = 7)
clf.fit(X_train , y_train)

clf.fit(X_test , y_test)


# In[94]:


#Using model to predict training

y_pred = clf.predict(X_train)
accuracy = accuracy_score(y_pred, y_train)
print('Training dataset accuracy is:', accuracy)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print('Testing dataset accuracy is:', accuracy)


# In[95]:


#Using model to predict test
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print('Test dataset accuracy is:', accuracy)

#Using model to predict train
y_pred = clf.predict(X_train)
accuracy = accuracy_score(y_pred, y_train)
print('Train dataset accuracy is:', accuracy)


# In[74]:


#Grid Search CV


# In[83]:


from sklearn.model_selection import GridSearchCV

#define search space for parameer grid
parameter_grid = {'n_neighbors': range(3,30)}



# In[99]:


#create learning model

knn_clf = KNeighborsClassifier()
clf = GridSearchCV (knn_clf, parameter_grid, scoring='accuracy', cv = 5)
clf.fit(X_train , y_train)


# In[100]:


cv_results = clf.cv_results_
mean_test_accuracy = cv_results['mean_test_score']
knn_params = cv_results['params']


# In[101]:


k_values = [param_dict['n_neighbors'] for param_dict in knn_params]

plt.figure(figsize=(9, 7))
plt.plot(k_values, mean_test_accuracy)


# In[91]:


# identify bes parameter

print('Best K value', clf.best_params_['n_neighbors'])
print('Accuracy' , clf.best_score_)


# In[53]:


The graph shows how the mean test scores vary with K values
and it might show a trend where the accuracy initially increases with K, 
then stabilizes or decreases after reaching an optimal value.


# 
