#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/smart-stats/ds4bio_book/blob/main/book/basic_regression_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Basic regression in pytorch

# In[225]:


import pandas as pd
import torch
import statsmodels.formula.api as smf
import statsmodels as sm
import seaborn as sns
import matplotlib.pyplot as plt

## Read in the data and display a few rows
dat = pd.read_csv("https://raw.githubusercontent.com/bcaffo/ds4bme_intro/master/data/oasis.csv")
dat.head(4)


# In[193]:


sns.scatterplot(dat['T2'], dat['PD'])


# In[226]:


fit = smf.ols('PD ~ T2', data = dat).fit()
fit.summary()


# In[228]:


# The in sample predictions
yhat = fit.predict(dat['T2'])

# Make sure that it's adding the intercept
#test = 0.3138 + dat['T2'] * 0.7832
#sns.scatterplot(yhat,test)

## A plot of the in sample predicted values
## versus the actual outcomes
sns.scatterplot(yhat, dat['PD'])
plt.plot([-1, 3], [-1, 3], linewidth=2)


# In[221]:


n = dat.shape[0]

## Get the y and x from 
xtraining = torch.from_numpy(dat['T2'].values)
ytraining = torch.from_numpy(dat['PD'].values)

## PT wants floats
xtraining = xtraining.float()
ytraining = ytraining.float()

## Dimension is 1xn not nx1
## squeeze the second dimension
xtraining = xtraining.unsqueeze(1)
ytraining = ytraining.unsqueeze(1)

## Show that everything is the right size
[xtraining.shape, 
 ytraining.shape,
 [n, 1]
 ]


# In[ ]:


## Show that linear regression is a pytorch 
model = torch.nn.Sequential(
    torch.nn.Linear(1, 1)
)

## MSE is the loss function
loss_fn = torch.nn.MSELoss(reduction='sum')

## Set the optimizer
## There are lots of choices
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

## Loop over iterations
for t in range(10000):

    ## Forward propagation
  y_pred = model(xtraining)
    
  ## the loss for this interation
  loss = loss_fn(y_pred, ytraining)

  #print(t, loss.item() / n)

  ## Zero out the gradients before adding them up 
  optimizer.zero_grad()
  
  ## Backprop
  loss.backward()
  
  ## Optimization step
  optimizer.step()


# In[229]:


ytest = model(xtraining).detach().numpy().reshape(-1)
sns.scatterplot(ytest, yhat)
plt.plot([-1, 3], [-1, 3], linewidth=2)


# In[215]:


for param in model.parameters():
  print(param.data)

