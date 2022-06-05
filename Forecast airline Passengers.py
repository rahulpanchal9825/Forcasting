#!/usr/bin/env python
# coding: utf-8

# Q-Forecast Airlines Passengers data set. Prepare a document for each model explaining 
# how many dummy variables you have created and RMSE value for each model. Finally which model you will use for 
# Forecasting.

# In[134]:


get_ipython().system('pip install category_encoders')


# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import category_encoders as ce
import statsmodels.formula.api as smf

from numpy import log


# In[136]:


dataset=pd.read_excel("E:\DATA SCIENCE\LMS\ASSIGNMENT\MY ASSIGNMENT\Forcasting\Airlines+Data.xlsx")
print(dataset)


# In[137]:


dataset.info()


# From the plot below, we can see that there is a Trend compoenent in th series. Hence, we now check for stationarity of the data

# In[138]:


dataset.rename(columns = {"Month": "date"}, inplace = True)
dataset


# In[139]:


print(dataset.shape)
dataset.isnull().sum()


# In[140]:


ds=dataset


# In[141]:


ds.describe(include='all')


# In[142]:


ds.info()


# In[143]:


ds.Passengers.plot()
# the data has an exponential curve with multiplicative seasonality


# In[144]:


ds_air = ds.copy()
ds_air.head(5)


# In[175]:


t_list = [x for x in range(1,len(ds)+1)]
ds_air['t'] = t_list

ds_air['t_square'] = ds_air['t']*ds_air['t']

#log transformation
ds_air['log_passengers'] = np.log10(ds.Passengers)
ds_air.head(15)


# In[176]:


ds_air['months'] = ds['date'].dt.month_name() # saves months name from date
ds_air['months'] = [x[0:3] for x in ds_air.months] # will take only first 3 chars of months like jan,feb
ds_air['year']   = ds['date'].dt.year
ds_air


# In[177]:


dummy = pd.DataFrame(pd.get_dummies(ds_air['months']))
dummy


# In[178]:


colsequence = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
dummy = dummy.reindex(columns=colsequence ) #sorts the column headings

ds_air = pd.concat([ds_air,dummy], axis=1)
ds_air


# In[179]:


heatmap_passengers = pd.pivot_table(data=ds_air,values='Passengers',index='year',
                                    columns='months',aggfunc='mean',fill_value=0)
plt.figure(figsize=(10,8))
sns.heatmap(data =heatmap_passengers , annot=True,fmt='g')


# In[180]:


plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x='months', y='Passengers', data=ds_air)
plt.subplot(212)
sns.boxplot(x='year', y='Passengers', data=ds_air)


# In[181]:


sns.lineplot(x='year',y='Passengers',data=ds_air)


# In[182]:


sns.lineplot(x='months',y='Passengers',data=ds_air)


# In[183]:


x_train = ds_air.head(76)
x_test  = ds_air.tail(20)
print(x_train.shape)
print(x_test.shape)


# In[184]:


x_test


# In[185]:


le_model         = smf.ols('Passengers~t',data=x_train).fit()
predicted_linear = pd.Series(le_model.predict(pd.DataFrame(x_test['t'])))
rmse_linear      = np.sqrt(np.mean(np.array(x_test['Passengers'])-np.array(predicted_linear))**2)
rmse_linear      = round(rmse_linear,2)
rmse_linear


# In[186]:


#Exponential

Exp      = smf.ols('log_passengers~t',data=x_train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(x_test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(x_test['log_passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp = round(rmse_Exp,2)
rmse_Exp


# In[187]:


#Quadratic 

Quad = smf.ols('Passengers~t+t_square',data=x_train).fit()
pred_Quad = pd.Series(Quad.predict(x_test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(x_test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad = round(rmse_Quad,2)
rmse_Quad


# In[188]:


#Additive seasonality 

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=x_train).fit()
pred_add_sea = pd.Series(add_sea.predict(x_test[['Jan','Feb','Mar','Apr','May','Jun','Jul',
                                               'Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(x_test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea = round(rmse_add_sea,2)
rmse_add_sea


# In[189]:


#Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',
                       data=x_train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(x_test[['Jan','Feb','Mar','Apr','May','Jun','Jul',
                                               'Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(x_test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[190]:


##Multiplicative Seasonality

Mul_sea = smf.ols('log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',
                  data = x_train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(x_test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(x_test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea = round(rmse_Mult_sea,2)
rmse_Mult_sea


# In[191]:


#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',
                      data = x_train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(x_test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(x_test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea = round(rmse_Mult_add_sea,2)
rmse_Mult_add_sea


# In[192]:


#comparing the results
model_list                = ["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea",
                                 "rmse_Mult_add_sea"]
rmse_val_list             = [rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,
                                 rmse_Mult_add_sea]

table_rmse                = pd.DataFrame(columns=['Model','RMSE Values'])
table_rmse['Model']       = model_list
table_rmse['RMSE Values'] = rmse_val_list

table_rmse.sort_values(by=['RMSE Values'])


# In[193]:


new_data = ds_air.tail(12)
new_data


# In[194]:


new_data = new_data.drop(columns=['Passengers','log_passengers','year'])


# In[195]:


#picked the model with the lowest RMSE value
#training the model on the entire dataset with rmse_Mult_add_sea
model_full = smf.ols('log_passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',
                      data = ds_air).fit()


# In[196]:


predicted_new = model_full.predict(new_data)
predicted_new


# In[197]:


new_data['forecasted_passengers'] = predicted_new
new_data


# In[2]:


plt.title('Passenger count for the year 2002')
sns.lineplot(x='months',y='forecasted_passengers',data=new_data)


# In[199]:


y=(new_data['forecasted_passengers'])


# In[200]:


y


# In[203]:


# back to our anitilog values for kmow numbers of passagers
z=(10**y)


# In[204]:


z

