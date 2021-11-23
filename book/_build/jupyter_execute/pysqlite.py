#!/usr/bin/env python
# coding: utf-8

# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smart-stats/ds4bio_book/HEAD)
# 
# # sqlite in python
# 
# An sqlite3 library ships with python. In this tutorial, we'll discuss how to utilize this library and read sqlite tables into pandas. With this, you can generalize to other python APIs to other databases. First, let's continue on with our work from the previous notebook. A nice little tutorial can be found [here](https://datacarpentry.org/python-ecology-lesson/09-working-with-sql/index.html). 

# In[8]:


import sqlite3 as sq3
import pandas as pd

con = sq3.connect("sql/opioid.db")
# cursor() creates an object that can execute functions in the sqlite cursor

sql = con.cursor()

for row in sql.execute("select * from county_info limit 5;"):
    print(row)

    
# you have to close the connection
con.close


# Let's read this dataset into pandas.

# In[13]:


con = sq3.connect("sql/opioid.db")

county_info = pd.read_sql_query("SELECT * from county_info", con)

# you have to close the connection
con.close

county_info.head

