#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/smart-stats/ds4bio_book/blob/main/book/functions.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# # Functions
# Writing functions are an important aspect of programming. Writing functions helps automate redundant tasks and create more reusable code. Defining functions in python is easy. Let's write a function that raises a number to a power. (This is unnecessary of course.) Don't forget the colon.

# In[1]:


def pow(x, n = 2):
  return x ** n

print(pow(5, 3))


# Note our function has a mandatory arugment, `x`, and an optional arugment, `n`, that takes the default value 2. Consider this example
# to think about how python evaluates function arguments. These are all the same.

# In[4]:


print(pow(3, 2))
print(pow(x = 3, n = 2))
print(pow(n = 2, x = 3))
#pow(n = 2, 3) this returns an error, the second position is n, but it's a named argument too


# You can look here, [https://docs.python.org/3/tutorial/controlflow.html](https://docs.python.org/3/tutorial/controlflow.html), to study the rules. It doesn't make a lot of sense to get to cute with your function calling arguments. I try to obey both the order and the naming. I argue that this is the way to go since usually functions are written with some sensible ordering of arguments and naming removes all doubt. Python has a special variable for variable length arguments. Here's an example.

# In[5]:


def concat(*args, sep="/"):
 return sep.join(args)  

print(concat("a", "b", "c"))
print(concat("a", "b", "c", sep = ":"))


# Lambda can be used to create short, unnamed functions. This has a lot of uses that we'll see later. 

# In[6]:


f = lambda x: x ** 2
print(f(5))


# Here's an example useage where we use lambda to make specific "raise to the power" functions.

# In[7]:


def makepow(n):
 return lambda x: x ** n

square = makepow(2)
print(square(3))
cube = makepow(3)
print(cube(2))

