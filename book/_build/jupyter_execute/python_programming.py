#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/smart-stats/ds4bio_book/blob/main/book/python_programming.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smart-stats/ds4bio_book/HEAD)
# 
# # Python programming
# Try the following requests input from a user using the command `input` then it tests the value of the input using an `if` statement. A couple of things to take note of. First, python uses the white space instead of inclosing the statement in braces or parentheses. Secondly, don't forget the colons after programming statements like `if`, `else`, `for`, `def` and so on. Otherwise, if/else statements work just like you'd expect. 

# In[1]:


# do this if you'd like to prompt for an input
# x = input("are you mean (y/n)? > ")
# Let's just assume the user input 'n'
x = 'n'
if x == 'y': 
 print("Slytherine!")
else:
 print("Gryffindor")


# Just to further describe white space useage in python, consider testing whether `statementA` is True. Below, `statementB` is executed as part of the if statement whereas `statementC` is outside of it because it's not indented. This is often considered an eye rolling aspect of the language, but I think it's nice in the sense that it bakes good code identation practices into the language.
# 
# ```
# ## Some more about white space
# if statementA:
#   statementB   # Executed if statementA is True
# statementC     # Executed regardless since it's not indented
# ```
# The generic structure of if statements in python are 
# ```
# if statement1 :
#  ...
# elif statement2 :
#  ...
# else 
#  ...
# ```
# Here's an example (note this is just equal to the statement `(a < 0) - (a > 0)`

# In[3]:


a = 5

if a < 0 :
  a = -1
elif a > 0 :
  a = 1
else :
  a = 0

print(a)


# `for` and `while` loops can be used for iteration. Here's some examples

# In[8]:


for i in range(4) :
 print(i)
 


# In[7]:


x = 4
while x > 0 :
 x = x - 1
 print(x)


# Note `for` loops can iterate over list-like structures.

# In[9]:


for w in 'word':
 print(w)


# The range function is useful for creating a structure to loop over. It creates a data type that can be iterated over, but isn't itself a list. So if you want a list out of it, you have to convert it.

# In[16]:


a = range(3)
print(a)
print(list(a))

