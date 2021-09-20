#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/bcaffo/ds4bme_intro/blob/master/notebooks/notebook5_a.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Linear separable models with SMF

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn as skl
import statsmodels.formula.api as smf
import statsmodels as sm

## this sets some style parameters
sns.set()

## Read in the data and display a few rows
dat = pd.read_csv("https://raw.githubusercontent.com/bcaffo/ds4bme_intro/master/data/oasis.csv")


# In[15]:


dat.head()


# In[19]:


#results = smf.ols('PD ~ FLAIR + T1 + T2  + FLAIR_10 + T1_10 + T2_10 + FLAIR_20', data = dat).fit()
results = smf.ols('PD ~ FLAIR + T1', data = dat).fit()
print(results.summary2())


# In[20]:


y1 = dat.PD
x1 = np.zeros(y.size) + 1
x2 = dat.FLAIR
x3 = dat.T1

def resid(x, y):
  return y - x * np.sum(x * y) / np.sum(x ** 2)

## Regress x1 out of everything
ey1x1 = resid(x1, y1)
ex2x1 = resid(x1, x2)
ex3x1 = resid(x1, x3)

## Regress the residual for x2 out of everything
ey1x1x2 = resid(ex2x1, ey1x1)
ex3x1x2 = resid(ex2x1, ex3x1)


np.sum(ex3x1x2 * ey1x1x2) / np.sum(ex3x1x2 ** 2)

