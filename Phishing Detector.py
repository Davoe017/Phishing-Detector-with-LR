#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[6]:


df = pd.read_csv(r"C:\Users\HP\Downloads\1553778303_phishing\phishing.txt")


# In[7]:


df.head()


# In[8]:


df.shape
df.isnull().sum()


# In[10]:


X = df.iloc[:,:-1].values
y = df.iloc[:,30].values


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)


# In[12]:


scaler =StandardScaler()
sclx_train = scaler.fit_transform(x_train)
sclx_test = scaler.fit_transform(x_test)


# In[13]:


LogReg = LogisticRegression(C=100, random_state=0)
LogReg.fit(sclx_train, y_train)


# In[16]:


lr_pred = LogReg.predict(sclx_test)


# In[15]:


LogReg.score(sclx_train, y_train)
LogReg.score(sclx_test, y_test)


# In[18]:


from sklearn.metrics import confusion_matrix
confusMat = confusion_matrix(y_test, lr_pred)
confusMat


# In[19]:


X = df.iloc[0:5,[6,14]].values
y = df.iloc[0:5,30].values


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)


# In[21]:


sclx_train2 = scaler.fit_transform(x_train)
sclx_test2 = scaler.fit_transform(x_test)


# In[23]:


LogReg2 = LogisticRegression(C=100, random_state=0)
LogReg2.fit(sclx_train2, y_train) 

lr_pred2 = LogReg2.predict(sclx_test2)


# In[24]:


ConfusMat2 = confusion_matrix(y_test, lr_pred2)


# In[25]:


X = df.iloc[0:13,[6,14]].values
y = df.iloc[0:13,30].values


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)


# In[28]:


sclx_train3 = scaler.fit_transform(x_train)
sclx_test3 = scaler.fit_transform(x_test)


# In[29]:


LogReg3 = LogisticRegression(C=100, random_state=0)
LogReg3.fit(sclx_train3, y_train)

lr_pred3 = LogReg3.predict(sclx_test3)


# In[31]:


confusMat3 = confusion_matrix(y_test, lr_pred3)


# In[ ]:




