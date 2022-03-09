#!/usr/bin/env python
# coding: utf-8

# # Image processing with pillow
# 
# Pillow is a fork of the Python Image Library (PIL) in order to make that library more friendly.

# In[29]:


import PIL
from PIL import Image


im = Image.open("assetts/felix.jpg")
im.size


# In[31]:


im.crop((500, 2000, 1700, 3000))

