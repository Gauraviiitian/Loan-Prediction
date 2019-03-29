#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[80]:


df = pd.read_csv("./train.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df['ApplicantIncome'].hist(bins=50)


# In[6]:


df.boxplot(column='ApplicantIncome') # there are some outliers clearly


# In[7]:


df.boxplot(column='ApplicantIncome', by = 'Education') # graduates are dominating and generating some outliers


# In[8]:


df['LoanAmount'].hist(bins=50)


# In[9]:


df.boxplot(column='LoanAmount') # there are some outliers in this variable too


# In[10]:


df.count() # print the count for non-nan values for each column


# ### Lets Impute NA values

# In[81]:


# df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

df.boxplot(column='ApplicantIncome', by = ['Education', 'Self_Employed']) # graduates are dominating and generating some outliers


# In[82]:


df['Self_Employed'].value_counts()


# In[83]:


df['Self_Employed'].fillna('No',inplace=True)


# In[84]:


table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)


# In[85]:


df.count()


# In[86]:


table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median) 
table


# In[87]:


# Define function to return value of this pivot_table 
def fage(x): 
    return table.loc[x['Self_Employed'],x['Education']] 
# Replace missing values 
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# In[88]:


df.count()['LoanAmount']


# In[89]:


df['LoanAmount_log'] = np.log(df['LoanAmount']) 
df['LoanAmount_log'].hist(bins=20)


# In[90]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome'] 
df['TotalIncome_log'] = np.log(df['TotalIncome']) 
df['LoanAmount_log'].hist(bins=20)


# In[91]:


import seaborn as sns


# In[92]:


sns.boxplot(df['ApplicantIncome'])
sns.despine()


# In[93]:


test = df.groupby(['Gender'])
test.describe()


# ## Lets fill all the missing values

# In[94]:


df.count()


# In[95]:


df['Gender'].fillna('Male', inplace=True)


# In[96]:


df.groupby(['Married']).count()


# In[97]:


df['Dependents'].fillna(value = 0, inplace=True)
df['Married'].fillna(value = 'No', inplace=True)
df['Loan_Amount_Term'].fillna(value = np.mean(df['Loan_Amount_Term']), inplace=True)
df['Credit_History'].fillna(value = 0, inplace=True)


# ## Let's impute our dataset and split it into train set and test set

# In[145]:


features = ['Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'LoanAmount', 'Credit_History', 'Property_Area', 'TotalIncome']
y = df['Loan_Status']
X = df[features]
y.head()


# In[147]:


X = pd.get_dummies(X)
y = pd.get_dummies(y)['Y']
y.head()


# In[148]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)


# In[154]:


test_y.head()


# # Now let's apply some model on our data using sklearn

# ## First apply logistic regression

# In[155]:


#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset # Create logistic regression object
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
# Train the model using the training sets and check score
clf.fit(train_X, train_y)


# In[156]:


#Equation coefficient and Intercept
print('Coefficient: \n', clf.coef_)
print('Intercept: \n', clf.intercept_)


# In[157]:


# Lets make predictions on our test data
preds = clf.predict(test_X)


# In[159]:


test_y.shape


# In[161]:


from sklearn.metrics import accuracy_score
accuracy_score(preds, test_y)


# In[ ]:




