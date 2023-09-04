#!/usr/bin/env python
# coding: utf-8

# Import required libraries

# In[80]:


import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Import the data(Stress and Strain)

# In[81]:


data  = pd.read_csv("Al6061_150.csv",encoding = "unicode_escape")


# In[82]:


data


# Data Preparation

# In[5]:


data.isnull().sum()


# In[68]:


data = data.drop([data.index[847]])


# In[23]:


data


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


sn.pairplot(data=data)


# Data Visualization

# In[73]:


plt.plot(data["Strain"],data["Stress_MPa"])
plt.title("Stress vs Strain")
plt.xlabel("Stress")
plt.ylabel("Strain")
plt.grid()
plt.show()


# Linear Regression

# In[64]:


row = data.shape[0]
col = data.shape[1]
data.shape


# In[83]:


#Linear Regression
import random
prop_line = random.randint(0,row)
for i in range(2,row):
    x = pd.DataFrame(data,loc[0:i,['Strain']])    
    y = pd.DataFrame(data,loc[0:i,['Stress_MPa']])
   
    model = LinearRegression().fit(x,y)
    y_new = model.predict(x) 
    
    r2 = round(r2_score(y,y_new),2)
    if r2>= 0.99:
        prop_line = i
        
prop_line


# In[75]:


#Proportional point
x1 = data.loc[[prop_line]].Strain
y1 = data.iloc[[prop_line]].Stress_MPa


# In[77]:


x = pd.DataFrame(data,loc[0:prop_line,['Strain']])    
y = pd.DataFrame(data,loc[0:prop_line,['Stress_MPa']])

model = LinearRegression().fit(x,y)
y_new = model.predict(x) 
    


# Visualization

# In[28]:


plt.scatter(data["Strain"],data["Stress_MPa"],label = "Stress vs Strain")
plt.plot(x,y_new,'r',label="Linear Regression")
plt.scatter(x1,y1,'r',label="Proportional point")
plt.title("Stress vs Strain")
plt.xlabel("Stress")
plt.ylabel("Strain")
plt.legend()
plt.grid()
plt.show()


# In[65]:


#Fracture point
x4 = data.loc[row-2].Strain
y4 = data.loc[row-2].Stress_MPa


# In[66]:


max_stress = data.iloc[data['Stress_MPa'].idxmax()]
x3 = max_stress['Strain']
y3 = max_stress['Stress_MPa']


# In[97]:


#Yield Point
data["Offset"] = data["Strain"]+0.002
data


# In[98]:


x = pd.DataFrame(data,loc[0:prop_line,['Strain']])    
y = pd.DataFrame(data,loc[0:prop_line,['Stress_MPa']])

model = LinearRegression().fit(x,y)
y_new = model.predict(x) 
    
xx = pd.DataFrame(data,loc[0:prop_line,['Offset']])    
yy = pd.DataFrame(data,loc[0:prop_line,['Stress_MPa']])

model = LinearRegression().fit(xx,yy)
yy_new = model.predict(x) 
    


# In[99]:


plt.scatter(data["Strain"],data["Stress_MPa"],label = "Stress vs Strain")
plt.plot(x,y_new,'r',label="Linear Regression")
plt.scatter(xx,yy,'g',label="Offset Linear Regression")
plt.title("Stress vs Strain")
plt.xlabel("Stress")
plt.ylabel("Strain")
plt.legend()
plt.grid()
plt.show()


# In[92]:


#Yield Point
for i in range(50,row):
    if(data.iloc[prop_line]['Offset'] <=data.iloc[i]['Strain']):
        x2 = data.loc[i].Strain
        y2 = data.loc[i].Stress_MPa
        break


# In[100]:


plt.subplots(figsize=[15,10])
plt.scatter(data["Strain"],data["Stress_MPa"],label = "Stress vs Strain")
plt.plot(x,y_new,'r',label="Linear Regression")
plt.scatter(xx,yy_new,'g',label="Offset Linear Regression")
plt.scatter(x1,y1,marker='*',s=150,label="Proportional Point")
plt.scatter(x2,y2,marker='*',s=150,label="Yield Point")
plt.scatter(x3,y3,marker='*',s=150,label="Ultimate Tensile Strength")           
plt.scatter(x4,y4,marker='*',s=150,label="Fracture Point")
plt.title("Stress vs Strain")
plt.xlabel("Stress")
plt.ylabel("Strain")
plt.legend()
plt.grid()
plt.show()


# In[101]:


print('Proportional limit:',data.iloc[prop_line]['Stress_MPa'],'MPa')
print('Yield Point:',y2,'MPa')
print('Ultimate Tensile Strength:',y3,'MPa')
print('Fracture Point:',y4,'MPa')


# In[102]:


print(round(r2_score(y,y_new),2))


# In[ ]:




