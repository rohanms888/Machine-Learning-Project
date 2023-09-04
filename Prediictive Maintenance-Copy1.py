#!/usr/bin/env python
# coding: utf-8

# In[40]:


#Import Required Libraries
import pandas as pd
import numpy as np


# In[39]:


#Data column names
col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']


# Load Training Data

# In[47]:


df_train_raw = pd.read_csv('PM_train.txt',sep = ' ',header = None)
df_train_raw


# In[48]:


df_train_raw.drop([26,27], axis=1, inplace = bool((1))
df_train_raw.head()


# In[41]:


df_train_raw.columns = col_names
df_train_raw.head()


# In[36]:


df_train_raw.decribe()


# In[33]:


df_train_raw.dtypes


# In[34]:


df_train_raw.isnull().sum()


# Load Truth Data

# In[35]:


df_train_raw = pd.read_csv('PM_train.txt',sep = ' ',header = None)
df_train_raw


# In[49]:


df_truth_raw.drop([1],axis = 1,inplace = bool(1))
df_truth_raw.head()
df_truth_raw.columns = ['ttf']
df_truth_raw.head()


# In[ ]:




