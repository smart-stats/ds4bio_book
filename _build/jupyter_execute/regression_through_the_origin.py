#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/bcaffo/ds4bme_intro/blob/master/notebooks/regression_through_the_origin.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Regression through the origin
# 
# In this notebook, we investigate a simple poblem where we'd like to use one scaled regressor to predict another. That is, let $Y_1, \ldots Y_n$ be a collection of variables we'd like to predict and $X_1, \ldots, X_n$ be predictors. Consider minimizing
# 
# $$
# l = \sum_i ( Y_i - \beta X_i)^2 = || Y - \beta X||^2.
# $$
# 
# Taking a derivative of $l$ with respect to $\beta$ yields 
# 
# $$
# l' = \sum_i 2 (Y_i - \beta X_i) X_i.
# $$
# 
# If we set this equal to zero and solve for beta we obtain the classic solution:
# 
# $$
# \frac{\sum_i Y_i X_i}{\sum_i X_i^2} = \frac{<Y, X>}{||X||^2}
# $$

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dat = pd.read_csv("https://raw.githubusercontent.com/bcaffo/ds4bme_intro/master/data/oasis.csv")
dat.head()


# In[3]:


x = dat.T2
y = dat.PD
plt.plot(x, y, 'o')


# In[4]:


x = x - np.mean(x)
y = y - np.mean(y)
plt.plot(x, y, 'o')


# In[5]:


b = sum(y * x) / sum(x ** 2 )
b


# In[6]:


plt.plot(x, y, 'o')
t = np.array([-1.5, 2.5])
plt.plot(t, t * b)

