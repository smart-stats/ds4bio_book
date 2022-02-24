#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/smart-stats/ds4bio_book/blob/main/book/regression_through_the_origin.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smart-stats/ds4bio_book/HEAD)

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
# l' = - \sum_i 2 (Y_i - \beta X_i) X_i.
# $$
# 
# If we set this equal to zero and solve for beta we obtain the classic solution:
# 
# $$
# \hat \beta = \frac{\sum_i Y_i X_i}{\sum_i X_i^2} = \frac{<Y, X>}{||X||^2}.
# $$
# 
# Note further, if we take a second derivative we get
# 
# $$
# l'' = \sum_i 2 x_i^2  
# $$
# 
# which is strictly positive unless all of the $x_i$ are zero (a case of zero variation in the predictor where regresssion is uninteresting). Regression through the origin is a very useful version of regression, but it's quite limited in its application. Rarely do we want to fit a line that is forced to go through the origin, or stated equivalently, rarely do we want a prediction algorithm for
# $Y$ that is simply a scale change of $X$. Typically, we at least also want an intercept. In the example that follows, we'll address this by centering the data so that the origin is the mean of the $Y$ and the mean of the $X$. As it turns out, this is the same as fitting the intercept, but we'll do that more formally in the next section.
# 
# First let's load the necessary packages.

# In[20]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Now let's download and read in the data.

# In[21]:


dat = pd.read_csv("https://raw.githubusercontent.com/bcaffo/ds4bme_intro/master/data/oasis.csv")
dat.head()


# It's almost always a good idea to plot the data before fitting the model.

# In[25]:


x = dat.T2
y = dat.PD
plt.plot(x, y, 'o')


# Now, let's center the data as we mentioned so that it seems more reasonable to have the line go through the origin. Notice here, the middle of the data, both $Y$ and $X$, is right at (0, 0). 

# In[26]:


x = x - np.mean(x)
y = y - np.mean(y)
plt.plot(x, y, 'o')


# Here's our slope estimate according to our formula.

# In[27]:


b = sum(y * x) / sum(x ** 2 )
b


# Let's plot it so to see how it did. It looks good. Now let's see if we can do a line that doesn't necessarily have to go through the origin.

# In[28]:


plt.plot(x, y, 'o')
t = np.array([-1.5, 2.5])
plt.plot(t, t * b)

