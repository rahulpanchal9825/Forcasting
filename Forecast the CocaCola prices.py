#!/usr/bin/env python
# coding: utf-8

# Q-Forecast the CocaCola prices data set. Prepare a document for each model explaining 
# how many dummy variables you have created and RMSE value for each model. Finally which model you will use for 
# Forecasting.

# In[65]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import category_encoders as ce
import statsmodels.formula.api as smf

from numpy import log


# In[66]:


dataset=pd.read_excel("E:\DATA SCIENCE\LMS\ASSIGNMENT\MY ASSIGNMENT\Forcasting\CocaCola_Sales_Rawdata.xlsx")
print(dataset)


# In[67]:


dataset.info()


# In[68]:


print(dataset.shape)
dataset.isnull().sum()


# In[69]:


ds=dataset


# In[70]:


ds.describe(include='all')


# In[71]:


ds.Sales.plot()
# the data has an exponential curve with multiplicative seasonality


# In[72]:


def separateQuarter(x):
    list_q = x.split('_')
    return list_q[0]

ds['quarters'] = ds['Quarter'].apply(separateQuarter)

dummy = pd.DataFrame(pd.get_dummies(ds['quarters']))
ds = pd.concat([ds,dummy], axis=1)
ds


# In[73]:


def caculateYear(x):
    items  = x.split('_')
    year   = items[1]
    finalyear = '19'+year
    return int(finalyear)

ds['year'] = ds['Quarter'].apply(caculateYear)
ds.head(5)


# In[74]:


t_list = [x for x in range(1,len(ds)+1)]
ds['t'] = t_list

ds['t_square'] = ds['t']*ds['t']

#log transformation
ds['log_sales'] = np.log10(ds.Sales)
ds.head(15)


# In[75]:


heatmap_ = pd.pivot_table(data=ds,values='Sales',index='year',
                                    columns='quarters',aggfunc='mean',fill_value=0)

plt.figure(figsize=(10,8))
sns.heatmap(heatmap_, annot=True, fmt='g')


# In[77]:


plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x='quarters', y='Sales', data=ds)
plt.subplot(212)
sns.boxplot(x='year', y='Sales', data=ds)


# In[78]:


sns.lineplot(x='year',y='Sales',data=ds)


# In[79]:


sns.lineplot(x='quarters',y='Sales',data=ds)


# In[80]:


x_train = ds.head(30)
x_test  = ds.tail(12)
print(x_train.shape)
print(x_test.shape)


# In[81]:


x_test


# In[82]:


le_model         = smf.ols('Sales~t',data=x_train).fit()
predicted_linear = pd.Series(le_model.predict(pd.DataFrame(x_test['t'])))
rmse_linear      = np.sqrt(np.mean(np.array(x_test['Sales'])-np.array(predicted_linear))**2)
rmse_linear      = round(rmse_linear,2)
rmse_linear


# In[83]:


#Exponential

Exp      = smf.ols('log_sales~t',data=x_train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(x_test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(x_test['log_sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp = round(rmse_Exp,2)
rmse_Exp


# In[84]:


#Quadratic 

Quad = smf.ols('Sales~t+t_square',data=x_train).fit()
pred_Quad = pd.Series(Quad.predict(x_test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(x_test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad = round(rmse_Quad,2)
rmse_Quad


# In[85]:


#Additive seasonality 

add_sea = smf.ols('Sales~Q1+Q2+Q3',data=x_train).fit()
pred_add_sea = pd.Series(add_sea.predict(x_test[['Q1','Q2','Q3']]))
rmse_add_sea = np.sqrt(np.mean((np.array(x_test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea = round(rmse_add_sea,2)
rmse_add_sea


# In[86]:


#Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Sales~t+t_square+Q1+Q2+Q3',data=x_train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(x_test[['Q1','Q2','Q3','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(x_test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad = round(rmse_add_sea_quad,2)
rmse_add_sea_quad


# In[87]:


#Multiplicative Seasonality

Mul_sea = smf.ols('log_sales~Q1+Q2+Q3',
                  data = x_train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(x_test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(x_test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea = round(rmse_Mult_sea,2)
rmse_Mult_sea


# In[88]:


#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_sales~t+Q1+Q2+Q3',data = x_train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(x_test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(x_test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea = round(rmse_Mult_add_sea,2)
rmse_Mult_add_sea


# In[89]:


#comparing the results
model_list                = ["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea",
                                 "rmse_Mult_add_sea"]
rmse_val_list             = [rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,
                                 rmse_Mult_add_sea]

table_rmse                = pd.DataFrame(columns=['Model','RMSE Values'])
table_rmse['Model']       = model_list
table_rmse['RMSE Values'] = rmse_val_list

table_rmse.sort_values(by=['RMSE Values'])


# ## Predict for new time period

# In[90]:


new_data = ds.iloc[36:40,:]
#new_data = ds_coke1.tail(2)
new_data


# In[91]:


new_data = new_data.drop(columns=['Sales','log_sales','year'])


# In[92]:


#picked the model with the lowest RMSE value
#training the model on the entire dataset with rmse_Mult_add_sea
model_full = smf.ols('log_sales~t++Q1+Q2+Q3',
                      data = ds).fit()


# In[93]:


predicted_new = model_full.predict(new_data)
predicted_new


# In[94]:


new_data['forecasted_Sales'] = predicted_new
new_data


# In[95]:


plt.title('Sales count for the year 1995')
sns.lineplot(x='quarters',y='forecasted_Sales',data=new_data)


# In[96]:


y=new_data['forecasted_Sales'] 


# In[102]:


# back to our anitilog values for kmow numbers of passagers
Forcastedsales=(10**y)


# In[122]:


y=pd.concat([Forcastedsales,new_data.drop(['forecasted_Sales'],axis=1)],axis=1)
y

