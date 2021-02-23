#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# In[2]:


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[3]:


train_len = len(trainset)
test_len = len(testset)
index = list(range(train_len))
print(train_len, test_len)


# In[4]:


#construct validation set (lets use 10 percent)
np.random.shuffle(index)
#number of blocks of data
split = int(0.5 * train_len)
set1_index = index[split:]
set2_index = index[:split]
#Need to use a dataloader to control batch size and also enable SGD
set1_loader = torch.utils.data.DataLoader(trainset, sampler = set1_index, batch_size = 12, num_workers = 10)
set2_loader = torch.utils.data.DataLoader(trainset, sampler = set2_index, batch_size = 12, num_workers = 10)
test_loader = torch.utils.data.DataLoader(testset)


# In[6]:


set1dataiter = iter(set1_loader)
set1images, set1labels = set1dataiter.next()
set2dataiter = iter(set2_loader)
set2images, set2labels = set2dataiter.next()


# In[7]:


class CNNBlock1(nn.Module):
    def __init__(self):
        super(CNNBlock1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        return x
cnnblock1 = CNNBlock1()


# In[8]:


class CNNBlock2(nn.Module):
    def __init__(self):
        super(CNNBlock2, self).__init__()
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 4, padding = 1)
    
    def forward(self, x):
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x),2))
        return x
cnnblock2 = CNNBlock2()


# In[9]:


class CNNBlock3(nn.Module):
    def __init__(self):
        super(CNNBlock3, self).__init__()
        self.conv5 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size = 2)
    
    def forward(self, x):
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x
cnnblock3 = CNNBlock3()


# In[10]:


class MLPBlock1(nn.Module):
    def __init__(self):
        super(MLPBlock1, self).__init__()
        self.fc1 = nn.Linear(15*15*128, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 15*15*128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
mlpblock1 = MLPBlock1()


# In[11]:


class MLPBlock2(nn.Module):
    def __init__(self):
        super(MLPBlock2, self).__init__()
        self.fc4 = nn.Linear(7*7*256, 256)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64,10)
    
    def forward(self, x):
        x = x.view(-1, 7*7*256)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
mlpblock2 = MLPBlock2()


# In[12]:


class MLPBlock3(nn.Module):
    def __init__(self):
        super(MLPBlock3, self).__init__()
        self.fc7 = nn.Linear(6*6*512, 256)
        self.fc8 = nn.Linear(256, 64)
        self.fc9 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 6*6*512)
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x
mlpblock3 = MLPBlock3()


# In[13]:


class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.cnnblock1 = cnnblock1
        self.cnnblock2 = cnnblock2
        self.cnnblock3 = cnnblock3
        self.mlpblock1 = mlpblock1
        self.mlpblock2 = mlpblock2
        self.mlpblock3 = mlpblock3
    
    def forward(self, x):
        x1 = self.cnnblock1(x)
        x2 = self.cnnblock2(x1)
        x3 = self.cnnblock3(x2)
        x4 = self.mlpblock1(x1)
        x5 = self.mlpblock2(x2)
        x6 = self.mlpblock3(x3)
        x7 = x4 + x5 + x6
        return F.log_softmax(x7, dim=1)

ensemblemodel = EnsembleModel()


# In[14]:


optimizer = optim.SGD(ensemblemodel.parameters(), lr = 0.05)


# In[15]:


criterion = nn.CrossEntropyLoss()


# In[16]:


for epoch in range(301):
    
    if epoch % 2 == 0:
        for param in cnnblock1.parameters():
            param.requires_grad_(False)
        
        for param in cnnblock2.parameters():
            param.requires_grad_(False)
        
        for param in cnnblock3.parameters():
            param.requires_grad_(False)
        
        for param in mlpblock1.parameters():
            param.requires_grad_(True)
            
        for param in mlpblock2.parameters():
            param.requires_grad_(True)
            
        for param in mlpblock3.parameters():
            param.requires_grad_(True)
        
        running_loss = 0.0
        for i, data in enumerate(set2_loader,0):
            inputs, set2labels = data
            
            optimizer.zero_grad()
            
            outputs = ensemblemodel(inputs)
            loss = criterion(outputs, set2labels)
            loss.backward()
            optimizer.step()
            
            #print stats
            running_loss += loss.item()
            if i%20 == 19:
                print('[%d,%5d] loss: %.3f' % (epoch+1, i + 1, running_loss / 20))
                running_loss = 0.0
                
    else:
        for param in cnnblock1.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock2.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock3.parameters():
            param.requires_grad_(True)
        
        for param in mlpblock1.parameters():
            param.requires_grad_(False)
            
        for param in mlpblock2.parameters():
            param.requires_grad_(False)
            
        for param in mlpblock3.parameters():
            param.requires_grad_(False)
        
        running_loss = 0.0
        
        for i, data in enumerate(set1_loader,0):
            inputs, set1labels = data
            
            optimizer.zero_grad()
            
            outputs = ensemblemodel(inputs)
            loss = criterion(outputs, set1labels)
            loss.backward()
            optimizer.step()
            
            #print stats
            running_loss += loss.item()
            if i%21 == 20:
                print('[%d,%5d] loss: %.3f' % (epoch+1, i + 1, running_loss / 21))
                running_loss = 0.0

print('Finished Training')


# In[17]:


correct_count, all_count = 0, 0
for inp,labels in test_loader:
  for i in range(len(labels)):
    with torch.no_grad():
        logps = ensemblemodel(inp)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


# In[14]:


class EnsembleModel2(nn.Module):
    def __init__(self):
        super(EnsembleModel2, self).__init__()
        self.cnnblock1 = cnnblock1
        self.cnnblock2 = cnnblock2
        self.cnnblock3 = cnnblock3
        self.mlpblock1 = mlpblock1
        self.mlpblock2 = mlpblock2
        self.mlpblock3 = mlpblock3
    
    def forward(self, x):
        x1 = self.cnnblock1(x)
        x2 = self.cnnblock2(x1)
        x3 = self.cnnblock3(x2)
        x4 = self.mlpblock1(x1)
        x5 = self.mlpblock2(x2)
        x6 = self.mlpblock3(x3)
        x7 = x4 + x5 + x6
        return F.log_softmax(x7, dim=1)

ensemblemodel2 = EnsembleModel2()


# In[16]:


optimizer2 = optim.SGD(ensemblemodel2.parameters(), lr = 0.05)


# In[18]:


for epoch in range(201):
    
    if epoch % 2 == 0:
        for param in cnnblock1.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock2.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock3.parameters():
            param.requires_grad_(True)
        
        for param in mlpblock1.parameters():
            param.requires_grad_(True)
            
        for param in mlpblock2.parameters():
            param.requires_grad_(True)
            
        for param in mlpblock3.parameters():
            param.requires_grad_(True)
        
        running_loss = 0.0
        for i, data in enumerate(set2_loader,0):
            inputs, set2labels = data
            
            optimizer2.zero_grad()
            
            outputs = ensemblemodel2(inputs)
            loss = criterion(outputs, set2labels)
            loss.backward()
            optimizer2.step()
            
            #print stats
            running_loss += loss.item()
            if i%20 == 19:
                print('[%d,%5d] loss: %.3f' % (epoch+1, i + 1, running_loss / 20))
                running_loss = 0.0
                
    else:
        for param in cnnblock1.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock2.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock3.parameters():
            param.requires_grad_(True)
        
        for param in mlpblock1.parameters():
            param.requires_grad_(True)
            
        for param in mlpblock2.parameters():
            param.requires_grad_(True)
            
        for param in mlpblock3.parameters():
            param.requires_grad_(True)
        
        running_loss = 0.0
        
        for i, data in enumerate(set1_loader,0):
            inputs, set1labels = data
            
            optimizer2.zero_grad()
            
            outputs = ensemblemodel2(inputs)
            loss = criterion(outputs, set1labels)
            loss.backward()
            optimizer2.step()
            
            #print stats
            running_loss += loss.item()
            if i%21 == 20:
                print('[%d,%5d] loss: %.3f' % (epoch+1, i + 1, running_loss / 21))
                running_loss = 0.0

print('Finished Training')


# In[19]:


correct_count, all_count = 0, 0
for inp,labels in test_loader:
  for i in range(len(labels)):
    with torch.no_grad():
        logps = ensemblemodel2(inp)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

