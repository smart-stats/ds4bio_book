#!/usr/bin/env python
# coding: utf-8

# # Webscraping
# 
# We'll need some packages to start, `requests`,  `beautifulsoup4` and `selenium`. Requesting elements from a static web page is very straightforward. Let's take an example by trying to grab and plot the table of multiple Olympic medalists from Wikipedia then create a barplot of which sports have the most multiple medal winners. 
# 
# First we have to grab the data from the url, then pass it to beautifulsoup4, which parses the html, then pass it to pandas. First let's import the packages we need.

# In[38]:


import requests as rq
import bs4
import pandas as pd


# We then need to read the web page into data.

# In[67]:


url = 'https://en.wikipedia.org/wiki/List_of_multiple_Olympic_gold_medalists'
page = rq.get(url)
## print out the first 200 characters
page.text[0:199]


# Now let's read the page into bs4. Then we want to find the tables in the page. We add the class and wikitable information to specify which tables that we want. If you want to find classes, you can use a web tool, like selectorgadget or viewing the page source.

# In[70]:


bs4page = bs4.BeautifulSoup(page.text, 'html.parser')
tables = bs4page.find('table',{'class':"wikitable"})


# Now we should take the html that we've saved, then read it into pandas. Fortunately, pandas has a `read_html` method. So, we convert our tables to strings then read it in. Since there's multiple tables, we grab the first one. 

# In[71]:


medals = pd.read_html(str(tables))[0]
## There's an empty row
medals = medals.dropna()
medals.head()


# Now we're in a position to build our plot. Let's look at the count of 4 or more medal winers by sport and games.

# In[72]:


medals[['Sport', 'Games']].value_counts().plot.bar()

