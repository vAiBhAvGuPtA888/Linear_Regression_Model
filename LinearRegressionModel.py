#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.read_csv("Transformed_Housing_Data2.csv")


# In[3]:


data.head()


# In[4]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Y = data['Sale_Price']
X = scaler.fit_transform(data.drop(columns = ['Sale_Price']))
X = pd.DataFrame(data = X,columns =data.drop(columns = ['Sale_Price']).columns )
X


# In[5]:


X.corr()


# In[ ]:





# In[6]:


k= X.corr()
z = [[str(i),str(j)] for i in k.columns for j in k.columns if(k.loc[i,j]>abs(5)&(i!=j))]
z,len(z)


# In[7]:


from statsmodels.stats.outliers_influence import variance_inflation_factor as vf
vif_data = X
VIF = pd.Series([vf(vif_data.values,i) for i in range(vif_data.shape[1])],index = vif_data.columns)


# In[8]:


def MC_remover(vif_data):
    vif = pd.Series([vf(vif_data.values,i) for i in range(vif_data.shape[1])],index = vif_data.columns)
    if(vif.max()>5):
        print(vif[vif==vif.max()].index[0], 'has been removed')
        vif_data = vif_data.drop(columns = [vif[vif==vif.max()].index[0]])
        return vif_data        
    else:
        print(' No multicolinearity')
        return vif_data


# In[9]:


for i in range(7):
    vif_data = MC_remover(vif_data)


# In[10]:


x = vif_data
y= data['Sale_Price']
from sklearn.model_selection import train_test_split as tst
x_train,x_test,y_train,y_test = tst(x,y,test_size = 0.3, random_state= 101)


# In[11]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
lr.fit(x_train,y_train)


# In[12]:


lr.coef_


# In[21]:


predictions = lr.predict(x_test)
lr.score(x_train,y_train)


# # Residual Plot

# In[24]:


residuals = predictions - y_test
residual_table = pd.DataFrame({'residuals':residuals,'predictions':predictions})
residual_table = residual_table.sort_values(by = 'predictions')


# In[25]:


z = [i for i in range(int(residual_table['predictions'].max()))]
k = [0 for i in range(int(residual_table['predictions'].max()))]


# In[32]:


plt.figure(dpi = 130 , figsize = (17,7))
plt.scatter(residual_table['predictions'],residual_table['residuals'],color = 'blue',s=2)
plt.plot(z,k,color = 'green',linewidth = 3,label = 'regression line')
plt.ylim(-800000,800000)
plt.xlabel('fitted points(ordered by predictions)')
plt.ylabel('residuals')
plt.title('Linear regression model - residual plot')
plt.legend()
plt.show()


# # Model Coefficients

# In[33]:


coefficients_table = pd.DataFrame({'column':x_train.columns,'coefficients':lr.coef_})
coefficient_table = coefficients_table.sort_values(by = 'coefficients')


# In[34]:


plt.figure(dpi = 130 , figsize = (8,7))
x = coefficient_table['column']
y = coefficient_table['coefficients']
plt.barh(x,y)
plt.xlabel('Coefficients')
plt.ylabel('variables')
plt.title('Normalized Coefficient plot')
plt.show()

