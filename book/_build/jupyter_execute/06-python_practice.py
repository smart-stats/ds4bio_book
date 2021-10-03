#!/usr/bin/env python
# coding: utf-8

# # Python in practice
# 
# The kind of programming we've seen so far in python isn't how typical data programmming in python goes. Instead, we tend to rely a lot of modules that add methods to our complex data science objects. Most python objects are class objects that come with a variety of convenient methods associated with them. If you're working in a good coding environment, then it should have some method autocompletion for your objects, which helps prevent typos and can speed up work. Let's look at methods associated with a list object. Note that some methods change the object itself while others return things without changing the object.

# In[1]:


pets = ['frogs', 'cats', 'dogs', 'hamsters']
print(pets)
pets.sort() #note this changes the pets object
print(pets)
pets.reverse()
print(pets)
pets.pop()
print(pets)
pets.append("horses")
print(pets)
print(pets.count("horses")) #counts the number of times the string horses is in the list


# A useful working example is working with imaginary numbers.

# In[2]:


x = 10 + 5j
print(x.real)
print(x.imag)
print(x.conjugate())


# Let's create our own version of a complex number, adapted from [here](https://docs.python.org/3/tutorial/classes.html).

# In[3]:


class mycomplex:
    def __init__(self, real, imag):
        self.r = real
        self.i = imag

    def conjugate(self): #note this modifies self
        self.i =  -self.i

    def print(self):
        print((self.r, self.i))

y = mycomplex(10,5)
y.print()
y.conjugate()
y.print()


# Let's now create a version that doesn't modify the object when we conjugate. 

# In[4]:


class mycomplex:
    def __init__(self, real, imag):
        self.r = real
        self.i = imag

    def conjugate(self): # note this doesn't modify self and returns a new object
        return(mycomplex(self.r, -self.i))

    def print(self):
        print((self.r, self.i))

y = mycomplex(10,5)
y.print()
z = y.conjugate()
y.print()
z.print()

