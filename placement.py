#!/usr/bin/env python
# coding: utf-8

# ## Placement Decision Tree

# In[88]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from IPython.display import Image
import graphviz
import pydotplus


# In[2]:


cd D:/Michigan Tech/Projects/Placement_DecisionTree


# In[3]:


# Import the data CSV file
data = pd.read_csv('Placement_Data_Full_Class.csv')
data.head(5)


# In[4]:


# Check null values
data.isnull().sum()


# Salary is the field that has 67 null values and we do not think it is a very useful factor so we will remove it.

# In[5]:


data = data.drop(columns=['salary', 'sl_no'])
categorical_features = data.select_dtypes(include=['object']).copy()
numerical_features = data.select_dtypes(exclude=['object']).copy()


# In[6]:


# Distribution plot for numerical features
i=1
plt.figure(figsize=(16,6)) 
for c in numerical_features.columns:
    plt.subplot(2,3,i)
    sns.distplot(numerical_features[c])
    i+=1


# In[7]:


# Bar plot for categorical features
i=1
plt.figure(figsize=(16,6)) 
for c in categorical_features.columns:
    plt.subplot(3,3,i)
    categorical_features[c].value_counts().plot(kind='bar')
    i+=1


# In[69]:


data_enc = pd.get_dummies(data, drop_first=True)
sns.heatmap(data.corr(method='pearson'))


# In[70]:


plt.subplot(121)
categorical_features['specialisation'].value_counts().plot(kind='bar')
plt.title('Specialisation')
plt.subplot(122)
categorical_features[categorical_features['status']=='Placed'].specialisation.value_counts().plot(kind='bar')
plt.title('Specialisation | Placed')


# In[65]:


data.ssc_b.unique()
for c in categorical_features.columns:
    if c != 'status':
        print('-'*50)
        print(c)
        print('-'*50)
        print(categorical_features[c].value_counts()*100/len(categorical_features))
        print('\n- GIVEN PLACED:')
        print(categorical_features[categorical_features.status=='Placed'][c].value_counts()*100/len(categorical_features[categorical_features.status=='Placed']))


# From the above table we can observe that Workex has a high correlation with the placement. Also, people with Marketing and Finance specialisation seems to have a higher posibility of getting placed. There is also a slight indication that males got placed more than females. However, the difference is minimal and we need to further investigation before any comment.

# In[ ]:


data.ssc_b.unique()
for c in categorical_features.columns:
    if c == 'gender':
        print('-'*50)
        print(c)
        print('-'*50)
        print(categorical_features[c].value_counts()*100/len(categorical_features))
        print('\n- GIVEN PLACED:')
        print(categorical_features[categorical_features.gender=='Placed'][c].value_counts()*100/len(categorical_features[categorical_features.status=='Placed']))


# In[71]:


sns.heatmap(data.corr())


# In[77]:


x = data.drop(columns='status')
y = pd.DataFrame(data['status'])


# # Decision Tree

# In[72]:


x_enc = pd.get_dummies(x, drop_first=True)
x_enc.head()


# In[107]:


print('Placed: {}, Not placed: {}'.format(sum(y.status=='Placed'), sum(y.status=='Not Placed')))


# In[109]:


sum(data[y.status=='Placed'].ssc_p<=56.44)


# In[110]:


clf = tree.DecisionTreeClassifier()
tr = clf.fit(x_enc,y)
fig = plt.figure(figsize=(90,30))
t=tree.plot_tree(tr, feature_names=x_enc.columns, class_names=['Not Placed', 'Placed'])
plt.savefig('tree.png')

