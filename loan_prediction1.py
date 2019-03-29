#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[3]:


df = pd.read_csv("./train.csv")


# ## Quick data exploration and Visualization

# In[4]:


df.head() # display first five rows


# ### print the number of non-nan values for each feature(column)
# There are many missing values. Missing value of a particular feature will be 614-(count of that feature)

# In[5]:


df.count()


# ### a brief description of dataset

# In[6]:


df.describe()


# ## Heatmap
# ### Correlation among various numerical features in the dataframe
# LoanAmount and ApplicantIncome are a bit(score=0.57) correlated, no other pair is correlated as such.

# In[35]:


sb.heatmap(df.corr(),annot=True)
plt.show()


# ### Maximum applicants are from semiurban areas

# In[8]:


prop_area = df['Property_Area'].value_counts()
print(prop_area)
prop_area.plot(kind="bar")


# ### 81% applicants are Male

# In[9]:


gend = df['Gender'].value_counts()
print(gend)
gend.plot(kind="bar")


# ### 65% applicants are married

# In[10]:


marr = df['Married'].value_counts()
print(marr)
marr.plot(kind="bar")


# ### 78% applicants are Graduate

# In[11]:


edu = df['Education'].value_counts()
print(edu)
edu.plot(kind="bar")


# ### Only 14% applicants are self employed

# In[12]:


self_emp = df['Self_Employed'].value_counts()
print(self_emp)
self_emp.plot(kind="bar")


# ### 57.6% applicants don't have dependents

# In[14]:


dep = df['Dependents'].value_counts()
print(dep)
dep.plot(kind="bar")


# ## Distribution Analysis
# ### ApplicantIncome has many outliers (mean=5403.459283) and it's positively skewed distribution

# In[15]:


df['ApplicantIncome'].hist(bins=50)


# In[52]:


sb.boxplot(df['ApplicantIncome'], orient="v") # there are some outliers clearly


# ### Most of the outliers are caused by Graduate applicants

# In[51]:


sb.boxplot(x='Education', y='ApplicantIncome', data=df) # there are some outliers in this variable too
sb.despine()


# ## graduate applicants who are self employed are having the maximum ApplicantIncome than others

# In[48]:


sb.boxplot(x='Education', y='ApplicantIncome', hue='Self_Employed', data=df)
sb.despine()


# ### LoanAmount is also a normal distribution having outliers (mean=146.412162) and it's a positive skewed distribution

# In[18]:


df['LoanAmount'].hist(bins=50)


# ### there are many outliers present

# In[41]:


sb.boxplot(df['LoanAmount'], orient="v") # there are some outliers in this variable too
sb.despine()


# ## Categorical variable analysis

# In[31]:


temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())['Loan_Status']


# ## This shows that the chances of getting a loan are eight-fold if the applicant has a valid credit history

# In[22]:


fig = plt.figure(figsize=(8,4)) 
ax1 = fig.add_subplot(121) 
temp1.plot(kind='bar')
ax1.set_xlabel('Credit_History') 
ax1.set_ylabel('Count of Applicants') 
ax1.set_title("Applicants by Credit_History")  
ax2 = fig.add_subplot(122) 
temp2.plot(kind='bar')
ax2.set_xlabel('Credit_History') 
ax2.set_ylabel('Probability of getting loan') 
ax2.set_title("Probability of getting loan by credit history") 


# In[23]:


temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status']) 
print(temp3)
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# ### There is no clear bias among male and female candidates

# In[32]:


temp4 = pd.crosstab(df['Gender'], df['Loan_Status']) 
print(temp4)
temp4.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# ## Graduate applicants are more likely to get loans than non Graduates

# In[36]:


temp5 = pd.crosstab(df['Education'], df['Loan_Status']) 
print(temp5)
temp5.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# ## Male applicants who have 1 Credit History are most likely to get the loan

# In[37]:


temp6 = pd.crosstab([df['Credit_History'], df['Gender']], df['Loan_Status']) 
print(temp6)
temp6.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# ## Graduate applicants having 1 credit history are most likely to get the loan

# In[38]:


temp7 = pd.crosstab([df['Credit_History'], df['Education']], df['Loan_Status'])
print(temp7)
temp7.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# ### Now let's make our predictions based on this. Got score 0.652777777777778

# In[25]:


test_data = pd.read_csv('./test.csv')

import random
preds = [['Loan_ID', 'Loan_Status']]
for i in range(test_data.shape[0]):
    r = random.random()
    tf0 = temp4['N'][0][0]/(temp4['N'][0][0] + temp4['Y'][0][0])
    tm0 = temp4['N'][0][1]/(temp4['N'][0][1] + temp4['Y'][0][1])
    tf1 = temp4['N'][1][0]/(temp4['N'][1][0] + temp4['Y'][1][0])
    tm1 = temp4['N'][1][1]/(temp4['N'][1][1] + temp4['Y'][1][1])
    
    if test_data['Gender'][i]=='Female' and test_data['Credit_History'][i]==0:
        if r>=tf0:
            preds.append([test_data['Loan_ID'][i], 'Y'])
        else:
            preds.append([test_data['Loan_ID'][i], 'N'])
    elif test_data['Gender'][i]=='Female' and test_data['Credit_History'][i]==1:
        if r>=tf1:
            preds.append([test_data['Loan_ID'][i], 'Y'])
        else:
            preds.append([test_data['Loan_ID'][i], 'N'])
    elif test_data['Gender'][i]=='Male' and test_data['Credit_History'][i]==0:
        if r>=tm0:
            preds.append([test_data['Loan_ID'][i], 'Y'])
        else:
            preds.append([test_data['Loan_ID'][i], 'N'])
    elif test_data['Gender'][i]=='Male' and test_data['Credit_History'][i]==1:
        if r>=tm1:
            preds.append([test_data['Loan_ID'][i], 'Y'])
        else:
            preds.append([test_data['Loan_ID'][i], 'N'])
    else:
        preds.append([test_data['Loan_ID'][i], 'N'])
print(len(preds))


# In[26]:


test_data.shape[0]


# In[27]:


preds = pd.DataFrame(preds)


# In[28]:


preds.to_csv('preds.csv', index=None, header=None)


# In[ ]:




