#!/usr/bin/env python
# coding: utf-8

# The goal this week is to pratice using Markdown to write descriptive text in notebooks and then look at Python data structures including the pandas module that supports data structures designed for handling the kind of data we'll be working with.  
# 
# There is probably too much work to complete here in the workshop session, but please use this notebook after the workshop to practice your Python.  Remember to commit your changes to git as you go and push back to Github when you are done.

# In[1]:


student_name = "Ravshan"
student_id = "48270326"
print(student_id)
print(student_name)


# ## Markdown Practice
# 
# Complete this section as per the instructions in the iLearn practical page.  Add the required cells below this one.

# ## Lists and Dictionaries
# 
# First we look at some built in Python data structures: lists and dictionaries. 
# 
# A list is a sequence of things, unlike strongly typed languages (Java, C#) a list can contain a mixture of different types - there is no type for a list of integers or a list of lists.   Here are some lists:

# In[37]:


ages = [12, 99, 51, 3, 55]
names = ['steve', 'jim', 'mary', 'carrie', 'zin']
stuff = [11, 'eighteen', 6, ['another', 'list']]
print(ages[0], names[0], stuff[0])
print(ages[2], names[2], stuff[2])
print(ages[1:], names[1:], stuff[1:])
for name in names:
    print(name)


# 1. write code to print the first and third elements of each list
# 2. write code to select and print everything except the first element of each list
# 3. write a for loop that prints each element of the 'names' list

# A dictionary is an associative array - it associates a value (any Python data type) with a key. The key is usually a string but can be any immutable type (string, number, tuple).  Here's some code that counts the occurence of words in a string.  It stores the count for each word in a dictionary using the word as a key. If the word is already stored in the dictionary, it adds one to the count, if not, it initialises the count to one.  
# 
# The second for loop iterates over the keys in the dictionary and prints one line per entry.
# 
# Modify this example to be a bit smarter:
# - make sure that punctuation characters are not included as parts of a word, be careful with hyphens - should they be included or not?
# - make the count use the lowercase version of a word, so that 'The' and 'the' are counted as the same word
# - **Challenge**: find the first and second most frequent words in the text
# - **Challenge**: take your code and write it as a function that takes a string and returns a list of words with their counts in order

# In[38]:


description = """This unit introduces students to the fundamental techniques and 
tools of data science, such as the graphical display of data, 
predictive models, evaluation methodologies, regression, 
classification and clustering. The unit provides practical 
experience applying these methods using industry-standard 
software tools to real-world data sets. Students who have 
completed this unit will be able to identify which data 
science methods are most appropriate for a real-world data 
set, apply these methods to the data set, and interpret the 
results of the analysis they have performed. """

count = dict()
for word in description.split():
    if word in count:
        count[word] += 1
    else:
        count[word] = 1
        
for word in count:
    print(word, count[word])


# ## Pandas Data Frames
# 
# [Pandas](https://pandas.pydata.org) is a Python module that provides some important data structures for Data Science work and a large collection of methods for data analysis. 
# 
# The two main data structures are the [Series]() and [DataFrame](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe).  
# 
# A Series is a one dimensional array of data, but unlike the Python list the data is indexed - the index is like the dictionary key, any immutable value like a number or string.  You can use the label to select elements from the series as well as positional values.  
# 
# A DataFrame is analogous to a spreadsheet - a two dimensional table of data with indexed rows and named columns. 
# 
# You should read up on these and follow the examples in the text.  Here are a few exercises to complete with data frames.

# You are given three csv files containing sample data.

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

ds1 = 'Downloads/ds1.csv'
ds2 = 'files/ds2.csv'
ds3 = 'files/ds3.csv'


# Write code below to read one of these data files into a pandas data frame and:
# - show the first few rows: .head
# - find the summary data for each column: .describe
# - select just those rows where the value of x and y is over 50
# - select the column 'x' and create a series
# - plot the 'x' series as a line graph
# - plot the dataframe as a scatterplot
# 
# Once you have the code for this, you can change the file you use for input of the data (ds2, ds3) and re-run the following cells to see the different output that is generated

# In[60]:


import pandas as pd
ds1 = '/Users/sdfd/Downloads/ds1.csv'
df = pd.read_csv(ds1)
print(df.head())
print(df.describe)
selected_rows = df[(df['x'] > 50) & (df['y'] > 50)]
print(selected_rows)
x_series = df['x']
print(x_series)
df.plot()
df.plot.scatter(x='x', y='y')


# ## Checkpoint
# 
# Congratulations! you have finished the required task for Week 2. Since you got this empty (without your code and output) notebook by downloading from iLearn, place this notebook (Workshop Week 2.ipynb) into your local copy of your Github repository (e.g. practical-workshops-sonitsingh)and commit your work with a suitable commit message and push your changes back to your Github repository. Show your tutor your updated Github repository to get your checkpoint mark.

# # Further Practice
# 
# If you finish this task you can practice more with pandas data frames by following the examples in the text, section 2.6.  The CSV file that they use in that section is available in the `files` directory of this repository as `educ_figdp_1_Data.csv`.   

# In[ ]:


edufile = 'files/educ_figdp_1_Data.csv'

