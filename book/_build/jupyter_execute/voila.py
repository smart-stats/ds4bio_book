#!/usr/bin/env python
# coding: utf-8

# # Jupyterwidgets and voila
# 
# A very nice client server web app solution is given by jupyter widgets. This is a simplified framework that allows for widgets and sliders embedded into jupyter notebooks. They also work in google colab. Once you have a working notebook, you can host it on a web site if you'd like using a platform called voila. Or, you can just distribute the notebook as a notebook. 
# 
# (Note, it doesn't work with the published version of the book, since the book is simply running on a web server, not a voila server instance. So, we'll just show you screenshots.)
# 
# You first need to install jupyter widgets  with pip or conda. Then, restart your runtime and you're off to the races. In the next chapter, we'll introduce a more fully featured client server framework called dash. If you just want a simple app, especially if you want to distribute it as a notebook, voila should be your goto. If you need a full web application, then use dash (or one of the other python web frameworks).

# In[2]:


import ipywidgets as widgets


# ## Sliders
# 
# There are integer sliders and float sliders. Let's first figure out integer sliders.

# In[3]:



a = widgets.IntSlider()
b = widgets.IntSlider()

display(a)
display(b)


# In[4]:


print([a.value, b.value])


# In[5]:


a1 = widgets.Checkbox(
    value=False,
    description='Check me',
    disabled=False,
    indent=False
)
a2 = widgets.Checkbox(
    value=False,
    description='No check me',
    disabled=False,
    indent=False
)
display(a1)
display(a2)


# In[6]:


print([a1.value, a2.value])


# ## Showing real time widget interactions

# In[7]:


a = widgets.IntSlider(description='a')
b = widgets.IntSlider(description='b')
c = widgets.IntSlider(description='c')

def f(a, b, c):
    print(a + b + c)

out = widgets.interactive_output(f, {'a': a, 'b': b, 'c': c})

widgets.HBox([widgets.VBox([a, b, c]), out])

