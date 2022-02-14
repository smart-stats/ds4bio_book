#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/smart-stats/ds4bio_book/blob/main/book/convnet_classifier_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smart-stats/ds4bio_book/HEAD)
# 
# # Convnet classifier example
# 
# ## Convnet example
# 
# In this exercise, we'll build an autoencoder to model cryptopunks. You might have heard of the recent NFT (non-fungible token) craze. Cryptopunks are example NFT assets that one can buy. As of this writing, the cheapest Cryptopunk is worth over $40,000 dollars. The punks each have attributes, like a mustache or hairstyle. We'll train a conv net to classify punks by attributes. First we'll need to download all of the cryptopunks, which are in a giant single image file. Then separate them into the individual punks. (There's probably an easier way to do this, but this wasn't that hard.) Then we'll build a tensorflow model of the punks.

# In[1]:


import urllib.request
import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Import the image of all of the cryptopunks.

# In[2]:



imgURL = "https://raw.githubusercontent.com/larvalabs/cryptopunks/master/punks.png"
urllib.request.urlretrieve(imgURL, "cryptoPunksAll.jpg")


# In[3]:


img = PIL.Image.open("cryptoPunksAll.jpg").convert("RGB")
img


# It looks like there's 100x100=10,000 crypto punks each one being a 24x24 (x3 color channels) image. 

# In[4]:


img.size


# Convert to a numpy array and visualize some. Here's punk 0.

# In[5]:


imgArray = np.asarray(img)
plt.imshow(imgArray[0 : 23, 0 : 23, :])
plt.xticks([])
plt.yticks([])


# Here's punks 0 : 24. You can double check that the reference image is filling by rows associated with the punk's index by looking at links like these (change the final number which is the punk's index):
# 
# *   https://www.larvalabs.com/cryptopunks/details/0
# *   https://www.larvalabs.com/cryptopunks/details/1
# 
# 

# In[6]:


#Plot out the first 25 punks
plt.figure(figsize=(10,10))
for i in range(25): 
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  a, b = (24 * i), (24 * (i + 1))
  plt.imshow(imgArray[0 : 24, a : b, :])


# Reorder the array. I couldn't get reshape to do this right, but I think this is a one-line command waiting to happen. See if you can figure out a better way. All images are 24x24x3 and there's 10,000 punks. (Supposedly, there will only ever be 10k punks.) When I did this for tensorflow, it required 24x24x3. However, torch wants 3x24x24.
# 

# In[7]:


finalArray = np.empty((10000, 3, 24, 24))
for i in range(100):
  for j in range(100):
    a, b = 24 * i, 24 * (i + 1)  
    c, d = 24 * j, 24 * (j + 1) 
    idx = j + i * (100)
    finalArray[idx,0,:,:] = imgArray[a:b,c:d,0]
    finalArray[idx,1,:,:] = imgArray[a:b,c:d,1]
    finalArray[idx,2,:,:] = imgArray[a:b,c:d,2]


# Let's normalize our array and split it into testing and training data.

# In[8]:


n = finalArray.shape[0]
trainFraction = .75
sample = np.random.uniform(size = n) < trainFraction
x_train = finalArray[ sample, :, :, :] / 255
x_test =  finalArray[~sample, :, :, :] / 255
[x_train.shape, x_test.shape]


# Github user geraldb created a [github repo](https://github.com/cryptopunksnotdead/punks.attributes) containing databases of punk attributes. However, they're stored in a collection of csv files. So, first let's see if we can download them programmatically and concatenate them.

# In[9]:


baseUrl = "https://raw.githubusercontent.com/cryptopunksnotdead/punks.attributes/master/original/"
for i in range(0,10000, 1000):
  url = baseUrl+str(i)+"-"+str(i + 999)+".csv"
  print(url)
  if (i == 0):
    dat = pd.read_csv(url)
  else :
    dat = pd.concat ([dat, pd.read_csv(url)], 
                      join = 'inner',
                     ignore_index = True)


# Let's see how we did

# In[10]:


dat.head()


# Let's double check to see if the punk has an earring. 
# 

# In[11]:


dat = dat.assign(earring = dat[' accessories'].str.contains('Earring').astype(float).to_list())
dat.head()


# Let's see the distribution of earrings.

# In[12]:


dat.earring.value_counts(normalize=True)


# Let's get our y values.

# In[13]:


y_train = dat.earring[sample].to_numpy()
y_test =  dat.earring[~sample].to_numpy()

## Need to have the extra dimension
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)


# ## Pytorch
# 
# OK, now let's follow along with [this example](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and train a classifier. However, they do a lot of the data organization ahead of time. So, first we have to convert our training data and testing data into pytorch tensors. Then convert them into a dataset format. Then, convert them into a dataloader. The dataloader is useful since it will do things like automate the batch creation for us.

# In[14]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms

trainDataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
trainloader = torch.utils.data.DataLoader(trainDataset, batch_size = 100, shuffle = False, num_workers = 1)

#Actually, not necessary, I found it's easier to just test out
#on the data as a tensor and there's no reason to convert the y test
#testDataset  = TensorDataset(torch.Tensor(x_test ), torch.Tensor(y_test ))
#testloader  = torch.utils.data.DataLoader(testDataset , batch_size = 100, shuffle = False, num_workers = 1)


# Let's use their CIFAR code to check some of our images and labels.

# In[15]:


dataiter = iter(trainloader)
images, labels = dataiter.next()


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()


# show images

plt.figure(figsize = (10, 10))
imshow(torchvision.utils.make_grid(images[11 : 19, :, :, :]))

# print labels
labels[11 : 19,0]


# In[16]:


## Here's the sort of things we can do with the dataloaders
## basically iterate over the batches and it gives us stuff in
## the right format. Of course, this loop does nothing.
for i, data in enumerate(trainloader, 0):
  # get the inputs; data is a list of [inputs, labels]
  inputs, labels = data
[inputs.shape, labels.shape]


# In[17]:


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        ## This has to be changed because the size
        ## of our inputs is different than the CFAR
        ## example. There's is 32x32 and ours is 24x24
        ## Also, I changed the rest of the network architecture
        ## here
        ## Finally, we only have one output.
        self.fc1 = nn.Linear(16 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
net = Net()


# In[18]:


print(net)


# Let's run the inputs through our NN. Note, it outputs 10 (our batch size) and 1 (the number of outcomes we have).

# In[19]:


net(inputs).shape


# In[20]:


import torch.optim as optim
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[21]:


for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
#        if i % 2000 == 1999:    # print every 2000 mini-batches
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 2000))
#            running_loss = 0.0

print('Finished Training')


# In[23]:


## Run the testing data through the NN
testout = net(torch.Tensor(x_test)).detach().numpy()

## Compare with the testing labels
from sklearn.metrics import accuracy_score, roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, testout)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

