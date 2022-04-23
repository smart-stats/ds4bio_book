#!/usr/bin/env python
# coding: utf-8

# # Numpy
# 
# Numpy is an essential core component of doing statistics in python. Numpy basically contains all of the basic mathematical functions that you need. Let's load in some data and work with it a bit.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dat = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
## Get Italy, drop everyrthing except dates, convert to long (unstack converts to tuple)
Italy = dat[dat['Country/Region'] == 'Italy'].drop(["Province/State", "Country/Region", "Lat", "Long"], axis=1).unstack()


# Let's create a numpy array from the counts and work with it a bit. First we'll take our data from Italy and convert it from the cumulative case counts to the daily case counts.

# In[2]:


## convert from tuple to array
X = np.asarray(Italy)  
## get case counts instead of cumulative counts
X = X[1 : X.size] - X[0 : (X.size - 1)]


# Let's get some basic statistical summaries. Note the default is that the standard deviations uses the formula
# $$
# \sqrt{\frac{1}{N} \sum_{i=1}^N (X_i - \bar X)^2}
# $$
# rather than
# $$
# \sqrt{\frac{1}{N-1} \sum_{i=1}^N (X_i - \bar X)^2}.
# $$
# To get the latter (the unbiased version), set `ddof=1`. Personally, I prefer $N$ as a divisor, though that's a minority opinion. (Between bias or variance of the standard deviation estimate, I'd rather rather have lower variance.)

# In[3]:


print("Mean         : " + str(np.round(X.mean(), 2))) 
print("Std (biased) : " + str(np.round(X.std() , 2)))


# Numpy has a linear algebra library. Let's calculate a distributed lag model using numpy (typically you would use this with regression software). A distributed lag model is of the form:
# $$
# Y_t = \alpha + \sum_{i=1}^p \beta_i Y_{t-i} + \epsilon_i
# $$
# First, let's create the lagged matrix considering 3 lags.

# In[4]:


## Create a matrix of three lagged versions
X = np.array([ Italy.shift(1), Italy.shift(2), Italy.shift(3)]).transpose()
## Add a vector of ones
itc = np.ones( (X.shape[0], 1) )
X = np.concatenate( (itc, X), axis = 1)

## Visualize the results
X[0 : 10,:]


# Let's get rid of the three NA rows.

# In[5]:


X = X[ 3 : X.shape[0], :]
np.any(np.isnan(X))


# In[6]:


## Create the Y vector
Y = np.array(Italy[ 3 : Italy.shape[0]])
[Y.shape, X.shape]


# The matrix formula for minimizing the least squares regression model,
# $$
# || Y - X \beta||^2
# $$
# is given by
# $$
# \hat \beta = (X' X)^{-1} X' Y
# $$
# Let's do this in numpy. Let's find the estimated regression coefficients using the formula above. We'll use the following functions
# 
# * `matmul(A,B)` is the matrix multiplication of `A` and `B`
# * `A.T` is the transpose of `A`, labeled above as $A'$
# * `inv(A)` is the matrix inverse of `A`, labeled above as $A^{-1}$ 

# In[7]:


np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)


# Of course, we don't tend to do things this way. If needed, we'd use lstsq.

# In[8]:


np.linalg.lstsq(X, Y, rcond = None)[0]


# Typically, we wouldn't do any of this for this problem, since high level regression models exist already. For exmaple, sklearn's linear_model module)

# In[14]:


from sklearn import linear_model

model = linear_model.LinearRegression(fit_intercept = False)
fit = model.fit(X, Y)
fit.coef_

