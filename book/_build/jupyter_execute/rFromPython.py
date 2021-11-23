#!/usr/bin/env python
# coding: utf-8

# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smart-stats/ds4bio_book/HEAD)
# 
# # R from python 
# 
# Python has an R api called `Rpy2`. You can install it with `conda install rpy2` or `pip install rpy2`. We'll just cover some really basic examples. 

# In[1]:


import rpy2
import rpy2.rinterface as ri
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

#ri.initr()


# The robjects sub-library contains the simplest variation of the interface. Here's an example of executing R code in a python session.

# In[2]:


z = ro.r('''
     x = matrix(1 : 10, 5, 2)
     y = matrix(11 : 20, 5, 2)
     x + y;
     ''')
print(z)


# You can then operate on this matrix in python. Here's an example where we import the plotting library and use it.

# In[16]:


base = ro.packages.importr('base')
base.rowSums(z)


# Here's an example of defining a function in R and using it in python.

# In[5]:


fishersz = ro.r('function(r) .5 * log((1 + r) / (1 - r))')
fishersz(.7)

