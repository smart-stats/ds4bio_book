#!/usr/bin/env python
# coding: utf-8

# # Medical mnist
# 
# The medical mnist data is a great resource for trying out AI models on medical data that is smaler in scale. Install it with
# 
# ```
# pip install medmnist
# ```

# In[1]:


from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator

data_flag = 'chestmnist'
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train', download = True)
test_dataset = DataClass(split='test', download = True)


# In[3]:



train_loader = data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)


# In[4]:


train_dataset.montage(length=20)

