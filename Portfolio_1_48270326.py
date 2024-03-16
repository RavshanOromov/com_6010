#!/usr/bin/env python
# coding: utf-8

# ### Analysis of an E-commerce Dataset
# 
# We have been provided with a combined e-commerce dataset. In this dataset, each user has the ability to post a rating and review for the products they purchased. Additionally, other users can evaluate the initial rating and review by expressing their trust or distrust.
# 
# This dataset includes a wealth of information for each user. Details such as their profile, ID, gender, city of birth, product ratings (on a scale of 1-5), reviews, and the prices of the products they purchased are all included. Moreover, for each product rating, we have information about the product name, ID, price, and category, the rating score, the timestamp of the rating and review, and the average helpfulness of the rating given by others (on a scale of 1-5).
# 
# The dataset is from several data sources, and we have merged all the data into a single CSV file named 'A Combined E-commerce Dataset.csv'. The structure of this dataset is represented in the header shown below.
# 
# | userId | gender | rating | review| item | category | helpfulness | timestamp | item_id | item_price | user_city|
# 
#     | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |  ---- |  ---- |  
#     
# #### Description of Fields
# 
# * __userId__ - the user's id
# * __gender__ - the user's gender
# * __rating__ - the user's rating towards the item
# * __review__ - the user's review towards the item
# * __item__ - the item's name
# * __category__ - the category of the item
# * __helpfulness__ - the average helpfulness of this rating
# * __timestamp__ - the timestamp when the rating is created
# * __item_id__ - the item's id
# * __item_price__ - the item's price
# * __user_city__ - the city of user's birth
# 
# Note that, a user may rate multiple items and an item may receive ratings and reviews from multiple users. The "helpfulness" is an average value based on all the helpfulness values given by others.
# 
# There are four questions to explore with the data as shown below.
# 
# 
# 
# <img src="data-relation.png" align="left" width="400"/>
# (You can find the data relation diagram on iLearn - Portfolio Part 1 resources - Fig1)
# 

# In[40]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = pd.read_csv('/Users/sdfd/Downloads/The E-commerce Dataset.csv')


#  #### Q1. Remove missing data
# 
# Please remove the following records in the csv file:
# 
#  * gender/rating/helpfulness is missing
#  * review is 'none'
# 
# __Display the DataFrame, counting number of Null values in each column, and print the length of the data__ before and after removing the missing data.  

# In[41]:


# Drop records where 'gender', 'rating', or 'helpfulness' is missing
df = df.dropna(subset=['gender', 'rating', 'helpfulness'])

# Drop records where the 'review' is 'none'
df = df[df['review'] != 'none']

# Displayed the DataFrame before and after removing missing data
print("DataFrame before removing missing data:")
print(df)

# Counted the number of null values in each column
null_counts = df.isnull().sum()
print("\nNumber of null values in each column:")
print(null_counts)

# Printed the length of the data before and after removing missing data
print("\nLength of the data before removing missing data:", len(df))


# #### Q2. Descriptive statistics
# 
# With the cleaned data in Q1, please provide the data summarization as below:
# 
# * Q2.1 total number of unique users, unique reviews, unique items, and unique categories
# * Q2.2 descriptive statistics, e.g., the total number, mean, std, min and max regarding all rating records
# * Q2.3 descriptive statistics, e.g., mean, std, max, and min of the number of items rated by different genders
# * Q2.4 descriptive statistics, e.g., mean, std, max, min of the number of ratings that received by each items
# 

# In[42]:


# Displayed the DataFrame before removing missing data
print("\033[1mLength of data before removing missing data: \033[0m", len (df))
print("\n\033[1mDataFrame before removing missing data: \033[0m")
print(df. head ())
# Removed records where gender, rating, or helpfulness is missing, and where the review is 'none'
df = df. dropna(subset=['gender', 'rating', 'helpfulness'])
df = df[df['review'] != 'none']
# Displayed the DataFrame after removing missing data
print("\n\033[1mLength of data after removing missing data: \033[0m", len (df))
print("\n\033[1mDataFrame after removing missing data: \033[0m")
print(df. head( ))
# Counted the number of Null values in each column
print("\n\033[1mNumber of Null values in each column: \033[0m" )
print(df. isnull() .sum())


# #### Q3. Plotting and Analysis
# 
# Please try to explore the correlation between gender/helpfulness/category and ratings; for instance, do female/male users tend to provide higher ratings than male/female users? Hint: you may use the boxplot function to plot figures for comparison (___Challenge___)
#     
# You may need to select the most suitable graphic forms for ease of presentation. Most importantly, for each figure or subfigure, please summarise ___what each plot shows___ (i.e. observations and explanations). Finally, you may need to provide an overall summary of the data.

# In[43]:


# Plotting for gender vs. ratings
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='rating', data = df)
plt.title('Gender vs. Ratings')
plt.xlabel('Gender')
plt.ylabel('Rating')
plt.show()

# Plotting for helpfulness vs. ratings
plt.figure(figsize=(10, 6))
sns.boxplot(x='helpfulness', y='rating', data = df)
plt.title('Helpfulness vs. Ratings')
plt.xlabel('Helpfulness')
plt.ylabel('Rating')
plt.show()

# Plotting for category vs. ratings
plt.figure(figsize=(14, 6))
sns.boxplot(x='category', y='rating', data=df)
plt.title('Category vs. Ratings')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.xticks(rotation=45, ha='right')
plt.show()


# #### Q4. Detect and remove outliers
# 
# We may define outlier users, reviews and items with three rules (if a record meets one of the rules, it is regarded as an outlier):
# 
# 1. reviews of which the helpfulness is no more than 2
# 2. users who rate less than 7 items
# 3. items that receives less than 11 ratings
# 
# Please remove the corresponding records in the csv file that involves outlier users, reviews and items. You need to follow the order of rules to perform data cleaning operations. After that, __print the length of the data__.

# In[44]:


# Removed records where the helpfulness of reviews is no more than 2
df = df[df['helpfulness'] > 2]

# Identifying users who rate less than 7 items and removing their reviews
user_counts = df['userId'].value_counts()
outlier_users = user_counts[user_counts < 7].index
df = df[~df['userId'].isin(outlier_users)]

# Identifying items that receive less than 11 ratings and removed their reviews
item_counts = df['item_id'].value_counts()
outlier_items = item_counts[item_counts < 11].index
df = df[~df['item_id'].isin(outlier_items)]

# Printed the length of the cleaned DataFrame
print("Length of the cleaned data:", len(df))

