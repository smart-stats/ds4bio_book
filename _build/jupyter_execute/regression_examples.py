#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/bcaffo/ds4bme_intro/blob/master/notebooks/regression_examples.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Linear models: a classic example

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[45]:


dat = pd.read_csv("https://raw.githubusercontent.com/bcaffo/ds4bme_intro/master/data/swiss.csv")
dat.head()


# In[49]:


y = dat.Fertility
x = dat.drop(['Region', 'Fertility'], axis=1)
fit = LinearRegression().fit(x, y)
yhat = fit.predict(x)
[fit.intercept_, fit.coef_]


# In[50]:


x2 = x
x2['Test'] = x2.Agriculture + x2.Examination
fit2 = LinearRegression().fit(x2, y)
yhat2 = fit2.predict(x2)


# In[51]:


plt.plot(yhat, yhat2)


# In[53]:


x3 = x2.drop(['Agriculture'], axis = 1)
fit3 = LinearRegression().fit(x3, y)
yhat3 = fit3.predict(x3)
plt.plot(yhat, yhat3)

