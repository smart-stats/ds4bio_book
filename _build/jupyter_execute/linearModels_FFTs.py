#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/bcaffo/ds4bme_intro/blob/master/notebooks/LinearModels_and_FFTs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Regression and FFTs
# 
# Recall regression through the origin. If $y$ and $x$ are $n$-vectors of the same length, the minimizer of
# $$
# ||y - \beta x ||^2
# $$
# is $\hat \beta = <x, y> / ||x||^2$. Note, if $||x|| = 1$ then the estimate is just $\hat \beta = <x, y>$. Now consider a second variable, $w$, such that $<x, w> = 0$ and $||w|| = 1$. Consider now the least squares model
# $$
# ||y - \beta x - \gamma w||^2.
# $$
# We argued that the best estimate for $\beta$ now first gets rid of $w$ be regressing it out of $y$ and $x$. So, consider that
# $$
# ||y - <w, y> w - \beta (x - <w, x> w)||^2 =
# ||y - <w, y> w - \beta x||^2. 
# $$
# Thus, now the best estimate of $\beta$ is
# $$
# <y - <w, y> w, x> = <y, x>.
# $$
# Or, in other words, if $x$ and $w$ are orthogonal then the coefficient estimate for $x$ with $w$ included is the same as the coefficient of $x$ by itself. This extends to more than two regressors. 
# 
# If you have a collection of $n$ mutually orthogonal vectors of norm one, they are called an orthonormal basis. For an orthonomal basis, 1. the coefficients are just the inner products between the regressors and the outcome and 2. inclusion or exclusion of other elemenents of the basis doesn't change a basis elements estimated coefficients.
# 
# It's important to note, that this works quite generally. For example, for complex numbers as well as real. So, for example, consider the possibility that $x$ is $e^{-2\pi i m k / n}$ for $m=0,\ldots, n-1$ for a particular value of $k$. Vectors like this are orthogonal for different values of $k$ and all have norm 1. We have already seen that the Fourier coefficient is 
# $$
# f_k = <y, x> = \sum_{m=0}^{n-1} y_m e^{-2\pi i m k / n} = 
# \sum_{m=0}^{n-1} y_m \cos(-2\pi m k / n) + i \sum_{m=0}^{n-1} y_m \sin(-2\pi m k / n) 
# $$
# where $y_m$ is element $m$ of $y$. Thus, the Fourier coefficients are exactly just least squares coefficients applied in the complex space.  Thus we have that 
# $$
# f_k = a_k + i b_k
# $$
# where $a_k$ and $b_k$ are the coefficients from linear models with just the sine and cosine terms.
# 

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[159]:


dat = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
dat.head()


# In[160]:


## Get Italy, drop everyrthing except dates, convert to long (unstack converts to tuple)
## Like an idiot, I'm using x in a different sense here than in the text, which I figured out well after I had done everything
x=dat[dat['Country/Region'] == 'Italy'].drop(["Province/State", "Country/Region", "Lat", "Long"], axis=1).unstack()
## convert from tuple to array
x = np.asarray(x)  
## get case counts instead of cumulative counts
x = x[1 : x.size] - x[0 : (x.size - 1)]
## get the first non zero entry
x =  x[np.min(np.where(x !=  0)) : x.size]
plt.plot(x)


# In[161]:


n = x.size
t = np.arange(0, n, 1)
lowess = sm.nonparametric.lowess
xhat = lowess(x, t, frac=.05,return_sorted=False)
plt.plot(x)
plt.plot(xhat)


# In[162]:


## We're interested in the residual
y = x - xhat
plt.plot(y)


# In[163]:



## Create 4 elements
## Orthonormal basis (note dividing by sqrt(n/2) makes them norm 1)
c5  = np.cos(-2 * np.pi * t * 5 / n  ) / np.sqrt(n /2)
c20 = np.cos(-2 * np.pi * t * 20 / n ) / np.sqrt(n /2)
s5  = np.sin(-2 * np.pi * t * 5  / n  )/ np.sqrt(n /2)
s20 = np.sin(-2 * np.pi * t * 20 / n  ) / np.sqrt(n /2)


# In[164]:


## Verify that they are orthonormal mean 0
[
 np.sum(c5),
 np.sum(c20),
 np.sum(s5),
 np.sum(s20),
 np.sum(c5 * c5),
 np.sum(c20 * c20),
 np.sum(s5 * s5),
 np.sum(s20 * s20),
 np.sum(c5 * s5),
 np.sum(c5 * s20),
 np.sum(c5 * c20),
 np.sum(s5 * s20),
]


# In[165]:


f = np.fft.fft(y)
w = np.fft.fftfreq(n)
ind = w.argsort()
f = f[ind] 
w = w[ind]
plt.plot(w, f.real**2 + f.imag**2)


# In[166]:


[
 np.sum(c5 * y) * np.sqrt(n / 2),
 np.sum(c20 * y) * np.sqrt(n / 2),
 np.sum(s5 * y) * np.sqrt(n / 2),
 np.sum(s20 * y) * np.sqrt(n / 2),
] 


# In[167]:


sreg = linear_model.LinearRegression()
x=np.c_[c5, c20, s5, s20]
fit = sreg.fit(x, y)
fit.coef_ * np.sqrt(n/2)


# In[168]:


x=np.c_[c5, s5]
fit = sreg.fit(x, y)
fit.coef_ * np.sqrt(n/2)


# In[169]:


test = np.where( np.abs(f.real / np.sum(c5 * y) / np.sqrt(n / 2) - 1) < 1e-5) 
[test, f.real[test], w[test], 5 / n]


# In[170]:


f.imag[test]

