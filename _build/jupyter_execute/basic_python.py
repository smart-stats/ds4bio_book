#!/usr/bin/env python
# coding: utf-8

# # Basics
# In this section, we will cover some basic python concepts. Python is an extremely quick language to learn, but like most programming languages, can take a long time to master. In this class, we'll focus on a different style of programming than typical software development, programming with data. This will but less of a burden on us to be expert software developers in python, but some amount of base language knowledge is unavoidable. So, let's get started learning some of the python programming basics. I'm going to assume you've programmed before in some language. If that isn't the case, consider starting with a basic programming course of study before trying this book.
# 
# A great resource for learning basic python is the python.org documentation [https://docs.python.org/3/tutorial/index.html](https://docs.python.org/3/tutorial/index.html). My favorite programming resource of all time is the "Learn X in Y" tutorials. Here's one for python [https://learnxinyminutes.com/docs/python/](https://learnxinyminutes.com/docs/python/).

# Some of the basic programming types in python are ints, floats, complex and Boolean.  The command `type` tells us which. Here's an example with 10 represented in 4 ways (int, float, string and complex)  and the logical value `True`. Note, we're using `print` to print out the result of the `type` command. If you're typing directly into the python command line (called the repl, for read, evaluate, print, loop), you won't need the print statements. But, if you're using a notebook you probably will.
# 
# If you want an easy repl environment to program in, try [https://replit.com/](https://replit.com/). For an easy notebook solution to try out, look at google colab, [https://colab.research.google.com](https://colab.research.google.com) .

# In[1]:


print(type(10))
print(type(10.0))
print(type(10+0j))
print(type("10"))
print(type(True))


# These types are our basic building blocks. There's some other important basic types that build on these. We'll cover these later, but to give you a teaser:

# In[2]:


print(type([1, 2]))
print(type((1, 2))) 
print(type({1, 2}))


# Types can be converted from one to another. For example, we might want to change our 10 into different types. Here's some examples of converting the integer 10 into a float. First, we use the `float` function. Next we define a variable `a` that takes the integer value 10, then use a method associated with `a` to convert the type. If you're unfamiliar with the second notation, don't worry about that now, you'll get very used to it as we work more in python.

# In[3]:


print(type(float(10)))
a = 10
print(type( a.__float__() ))


# Python's repl does all of the basic numerical calculations that you'd like. It does dynamic typing so that you can do things like add ints and floats. Here we show the basic operators, and note `#` is a comment in python.

# In[4]:


print(10 + 10.0) ## addition, in this case an int and float
print(10 ** 2)   ## exponent
print(10 * 2)    ## multiplication
print(10 / 2)    ## division
print(10.1 // 2) ## integer division
print(10.1 % 2)  ## modulo


# Strings are easy to work with in python. Type `print("Hello World")` in the repl just to get that out of the way. Otherwise, `+` concatenates strings and brackets reference string elements. Here's some examples. Remember counting starts at 0 and negative numbers count from the back.

# In[5]:


word = "ds4bio"
print(word + " is great") #concatenation 
print(word[0])
print(word[-1])


# The strings `True` and `False` are reserved for the respective Boolean values. The operators `==`, `>`, `<`, `>=`, `<=` and `!=` are the testing operators while `and`, `or` and `is` are Boolean operators. Here are some examples.  

# In[6]:


a = 5 < 4  # sets a to False
b = 5 == 5 # sets b to True
print(a or b)
print(a and b)


# Don't define new variables called `TRUE` or `FALSE` or `tRuE` or `FaLsE`, or whatever, even though you can. Just get used to typing True and False the way python likes and don't use similar named things for other reasons.

# ### Data structures
# Python has some more advanced data structures that build on its primitive types. 
# 
# * Lists:  ordered collections of objects
# * Sets: like lists but only have unique elements
# * Tuples: like lists, but not mutable, i.e. need to create a new one to modify
# * Dictionaries: named elements that you can reference by their name rather than position
# 
# First, let's look at some list operations.
# 

# In[7]:


dat = [1, 4, 8, 10] # define a list
print(dat[0])       # reference an element
print(dat[2 : 4])   # reference elements
print(dat[2 : ]) 
print(dat[:2])
dat2 = [dat, dat]        # creating a list of lists
print(dat2)
print(dat2[1][2])        # referencing an item in a nested list
dat3 = [dat2, "string1"] # mixed types
print(dat3)
dat4 = dat + dat         # list concatenation
print(dat4)


# Now, let's look at dictionaries. 

# In[8]:


dict = {"a" : 1, "b" : 2} # Create a dictionary of two elements named a and b taking values 1 and 2 respectively
print(dict)
print(dict['a'])          # reference the element named a

