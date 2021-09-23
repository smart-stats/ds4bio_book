#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/bcaffo/ds4bme_intro/blob/master/notebooks/notebook5_a.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Interpretation of linear regression coefficients.
# 
# The module `statsmodels` gives a  particularly convenient R-like formula approach to fitting linear models.
# It allows for a model specification of the form `outcome ~ predictors`. We give an example below.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn as skl
import statsmodels.formula.api as smf

## this sets some style parameters
sns.set()

## Read in the data and display a few rows
dat = pd.read_csv("https://raw.githubusercontent.com/bcaffo/ds4bme_intro/master/data/oasis.csv")


# In[2]:


results = smf.ols('PD ~ FLAIR + T1 + T2  + FLAIR_10 + T1_10 + T2_10 + FLAIR_20', data = dat).fit()
print(results.summary2())


# The interpretation of the FLAIR coefficient is as follows. We estimate an expected 0.0160 decrease in proton density per 1 unit change in FLAIR - *with all of the remaining model terms held constant*. The latter statements is important to remember. That is, it's improtant to remember that coefficients are adjusted for the linear associations with other variables. One way to think about this is that both the PD and FLAIR variables have had the linear association with the other variables removed before relating them. The same is true for the other variables. The coefficient for T1 is interpreted similarly, it's the relationship between PD and T1 where the linear associations with the other variables had been removed from them both. Let's show this for the FLAIR variable.

# In[3]:



# Model for PD with FLAIR removed
dat['PD_adjusted'] = smf.ols('PD ~ T1 + T2  + FLAIR_10 + T1_10 + T2_10 + FLAIR_20', data = dat).fit().resid
# Model for FLAIR 
dat['FLAIR_adjusted'] = smf.ols('FLAIR ~ T1 + T2  + FLAIR_10 + T1_10 + T2_10 + FLAIR_20', data = dat).fit().resid


out = smf.ols('PD_adjusted ~ FLAIR_adjusted', data = dat).fit()
print(out.summary2())


# Notice that the coefficient is exactly the same (-0.0160). This highlights how linear regression "adjusts" for the other variables. It removes the linear association with them from both the explantory and outcome variables.
