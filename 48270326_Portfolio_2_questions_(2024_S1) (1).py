#!/usr/bin/env python
# coding: utf-8

# ## Analysis of an E-commerce Dataset Part 2

# The goal of the second analysis task is to train linear regression models to predict users' ratings towards items. This involves a standard Data Science workflow: exploring data, building models, making predictions, and evaluating results. In this task, we will explore the impacts of feature selections and different sizes of training/testing data on the model performance. We will use another cleaned combined e-commerce sub-dataset that **is different from** the one in “Analysis of an E-commerce Dataset” task 1.

# ### Import Cleaned E-commerce Dataset
# The csv file named 'cleaned_ecommerce_dataset.csv' is provided. You may need to use the Pandas method, i.e., `read_csv`, for reading it. After that, please print out its total length.

# In[1]:


import pandas as pd
df = pd.read_csv('/Users/sdfd/Desktop/cleaned_ecommerce.csv')


# ### Explore the Dataset
# 
# * Use the methods, i.e., `head()` and `info()`, to have a rough picture about the data, e.g., how many columns, and the data types of each column.
# * As our goal is to predict ratings given other columns, please get the correlations between helpfulness/gender/category/review and rating by using the `corr()` method.
# 
#   Hints: To get the correlations between different features, you may need to first convert the categorical features (i.e., gender, category and review) into numerial values. For doing this, you may need to import `OrdinalEncoder` from `sklearn.preprocessing` (refer to the useful exmaples [here](https://pbpython.com/categorical-encoding.html))
# * Please provide ___necessary explanations/analysis___ on the correlations, and figure out which are the ___most___ and ___least___ corrleated features regarding rating. Try to ___discuss___ how the correlation will affect the final prediction results, if we use these features to train a regression model for rating prediction. In what follows, we will conduct experiments to verify your hypothesis.

# In[2]:


print(df.columns)


# In[17]:


# Explore the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nSummary information about the dataset:")
print(df.info())

from sklearn.preprocessing import OrdinalEncoder

# Initialize the encoder
encoder = OrdinalEncoder()

# Encode categorical features
df[['gender', 'category']] = encoder.fit_transform(df[['gender', 'category']])

# Calculate correlations
correlations = df.corr()

# Correlation with the target variable 'rating'
rating_correlations = correlations['rating']

# Sort the correlations
rating_correlations_sorted = rating_correlations.abs().sort_values(ascending=False)

print("Correlation with 'rating':")
print(rating_correlations_sorted)


# ### Split Training and Testing Data
# * Machine learning models are trained to help make predictions for the future. Normally, we need to randomly split the dataset into training and testing sets, where we use the training set to train the model, and then leverage the well-trained model to make predictions on the testing set.
# * To further investigate whether the size of the training/testing data affects the model performance, please random split the data into training and testing sets with different sizes:
#     * Case 1: training data containing 10% of the entire data;
#     * Case 2: training data containing 90% of the entire data.
# * Print the shape of training and testing sets in the two cases.

# In[18]:


from sklearn.model_selection import train_test_split

# Case 1: Training data containing 10% of the entire data
X_train_case1, X_test_case1, y_train_case1, y_test_case1 = train_test_split(df.drop(columns=['rating']), df['rating'], test_size=0.9, random_state=42)

# Print the shape of training and testing sets for Case 1
print("Case 1:")
print("Training data shape:", X_train_case1.shape, y_train_case1.shape)
print("Testing data shape:", X_test_case1.shape, y_test_case1.shape)

# Training data containing 90% of the entire data
X_train_case2, X_test_case2, y_train_case2, y_test_case2 = train_test_split(df.drop(columns=['rating']), df['rating'], test_size=0.1, random_state=42)

# Print the shape of training and testing sets for Case 2
print("\nCase 2:")
print("Training data shape:", X_train_case2.shape, y_train_case2.shape)
print("Testing data shape:", X_test_case2.shape, y_test_case2.shape)


# ### Train Linear Regression Models with Feature Selection under Cases 1 & 2
# * When training a machine learning model for prediction, we may need to select the most important/correlated input features for more accurate results.
# * To investigate whether feature selection affects the model performance, please select two most correlated features and two least correlated features from helpfulness/gender/category/review regarding rating, respectively.
# * Train four linear regression models by following the conditions:
#     - (model-a) using the training/testing data in case 1 with two most correlated input features
#     - (model-b) using the training/testing data in case 1 with two least correlated input features
#     - (model-c) using the training/testing data in case 2 with two most correlated input features
#     - (model-d) using the training/testing data in case 2 with two least correlated input features
# * By doing this, we can verify the impacts of the size of traing/testing data on the model performance via comparing model-a and model-c (or model-b and model-d); meanwhile the impacts of feature selection can be validated via comparing model-a and model-b (or model-c and model-d).    

# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Feature Selection
correlation_matrix = df[['helpfulness', 'gender', 'category', 'review', 'rating']].corr()

# Get the two most correlated and two least correlated features with 'rating'
most_correlated_features = correlation_matrix.nlargest(2, 'rating')['rating'].index
least_correlated_features = correlation_matrix.nsmallest(2, 'rating')['rating'].index

# Split the dataset for Case 1 (10% training data)
X_train_case1, X_test_case1, y_train_case1, y_test_case1 = train_test_split(df[most_correlated_features], df['rating'], test_size=0.9, random_state=42)

# Split the dataset for Case 2 (90% training data)
X_train_case2, X_test_case2, y_train_case2, y_test_case2 = train_test_split(df[most_correlated_features], df['rating'], test_size=0.1, random_state=42)

# Train linear regression models
model_a_case1 = LinearRegression()
model_a_case1.fit(X_train_case1, y_train_case1)

model_b_case1 = LinearRegression()
model_b_case1.fit(X_train_case1, y_train_case1)

model_c_case2 = LinearRegression()
model_c_case2.fit(X_train_case2, y_train_case2)

model_d_case2 = LinearRegression()
model_d_case2.fit(X_train_case2, y_train_case2)

# Step 4: Evaluate the models
pred_a_case1 = model_a_case1.predict(X_test_case1)
mse_a_case1 = mean_squared_error(y_test_case1, pred_a_case1)

pred_b_case1 = model_b_case1.predict(X_test_case1)
mse_b_case1 = mean_squared_error(y_test_case1, pred_b_case1)

pred_c_case2 = model_c_case2.predict(X_test_case2)
mse_c_case2 = mean_squared_error(y_test_case2, pred_c_case2)

pred_d_case2 = model_d_case2.predict(X_test_case2)
mse_d_case2 = mean_squared_error(y_test_case2, pred_d_case2)

# Print mean squared errors for each model
print("Model-a (Case 1) MSE:", mse_a_case1)
print("Model-b (Case 1) MSE:", mse_b_case1)
print("Model-c (Case 2) MSE:", mse_c_case2)
print("Model-d (Case 2) MSE:", mse_d_case2)


# ### Evaluate Models
# * Evaluate the performance of the four models with two metrics, including MSE and Root MSE
# * Print the results of the four models regarding the two metrics

# In[20]:


import numpy as np

# Calculate MSE for each model
mse_a_case1 = mean_squared_error(y_test_case1, pred_a_case1)
mse_b_case1 = mean_squared_error(y_test_case1, pred_b_case1)
mse_c_case2 = mean_squared_error(y_test_case2, pred_c_case2)
mse_d_case2 = mean_squared_error(y_test_case2, pred_d_case2)

# Calculate RMSE for each model
rmse_a_case1 = np.sqrt(mse_a_case1)
rmse_b_case1 = np.sqrt(mse_b_case1)
rmse_c_case2 = np.sqrt(mse_c_case2)
rmse_d_case2 = np.sqrt(mse_d_case2)

# Print the results
print("Model-a (Case 1) - MSE:", mse_a_case1, "RMSE:", rmse_a_case1)
print("Model-b (Case 1) - MSE:", mse_b_case1, "RMSE:", rmse_b_case1)
print("Model-c (Case 2) - MSE:", mse_c_case2, "RMSE:", rmse_c_case2)
print("Model-d (Case 2) - MSE:", mse_d_case2, "RMSE:", rmse_d_case2)


# ### Visualize, Compare and Analyze the Results
# * Visulize the results, and perform ___insightful analysis___ on the obtained results. For better visualization, you may need to carefully set the scale for the y-axis.
# * Normally, the model trained with most correlated features and more training data will get better results. Do you obtain the similar observations? If not, please ___explain the possible reasons___.

# In[21]:


import matplotlib.pyplot as plt

# Create lists to hold the metrics and model names
model_names = ['Model-a (Case 1)', 'Model-b (Case 1)', 'Model-c (Case 2)', 'Model-d (Case 2)']
mse_values = [mse_a_case1, mse_b_case1, mse_c_case2, mse_d_case2]
rmse_values = [rmse_a_case1, rmse_b_case1, rmse_c_case2, rmse_d_case2]

# Set the scale for the y-axis
max_mse = max(mse_values)
max_rmse = max(rmse_values)

plt.figure(figsize=(10, 6))

# Plot MSE values
plt.subplot(1, 2, 1)
plt.bar(model_names, mse_values, color='skyblue')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.ylim(0, max_mse + 0.1*max_mse)  # Set y-axis limits

# Plot RMSE values
plt.subplot(1, 2, 2)
plt.bar(model_names, rmse_values, color='lightgreen')
plt.title('Root Mean Squared Error (RMSE)')
plt.ylabel('RMSE')
plt.ylim(0, max_rmse + 0.1*max_rmse)  # Set y-axis limits

plt.tight_layout()
plt.show()


# ### Data Science Ethics
# *Please read the following examples [Click here to read the example_1.](https://www.vox.com/covid-19-coronavirus-us-response-trump/2020/5/18/21262265/georgia-covid-19-cases-declining-reopening) [Click here to read the example_2.](https://viborc.com/ethics-and-ethical-data-visualization-a-complete-guide/)
# 
# *Then view the picture ![My Image](figure_portfolio2.png "This is my image")
# Please compose an analysis of 100-200 words that evaluates potential ethical concerns associated with the infographic, detailing the reasons behind these issues.
# 

# In[ ]:


# The article discusses ethical concerns regarding how Covid-19 data is reported in Georgia.
# One major issue is that the data can be presented in a misleading way,
# such as using graphs that aren't organized in chronological order,
# which makes it harder to understand the true trends.
# There are also problems with how data is shown on maps,
# where changing the scale can make it seem like there are fewer cases than there actually are. 
# Additionally, there are discrepancies between testing data reported by the CDC and what states report,
# which makes it hard for everyone to trust the data and make good decisions about public health. 
# Overall, the article highlights the importance of being clear, accurate,
# and honest when reporting data during health emergencies so that everyone can understand what's going on and make the best choices.

