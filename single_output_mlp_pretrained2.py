#!/usr/bin/env python
# coding: utf-8

#import key packages
import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


#set seed for reproducibility, could do extra for cuda but would slow performance
random.seed(12345)
torch.manual_seed(12345)
np.random.seed(12345)
device = torch.device("cuda:0")
device1 = torch.device("cpu")



#downloading the data
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


#set some parameters here

learnrate = 0.03
train1 = 0.1*train_len
train2 = 0.1*train_len
valsiz = 0.1*train_len
OPTIM = 'sgd'
activation = 'ReLU'
nepochs = 201


# shuffle data for "randomness" 
np.random.shuffle(index)


#Generate training sets, validations sets with no overlap together with test sets
split = int(0.1*train_len)
set1_index = index[0:split]
set2_index = index[split:2*split]
val_index = index[2*split:3*split]
index2 = list(range(test_len))
np.random.shuffle(index2)
split2 = int(0.03 * test_len)
test_index = index2[0:split2]
print(len(set1_index))
print(len(set2_index))
print(len(test_index))
set1_loader = torch.utils.data.DataLoader(trainset, sampler = set1_index, batch_size = 10, num_workers = 8)  #batch size 10 because when it was 100 had memory issues
set2_loader = torch.utils.data.DataLoader(trainset, sampler = set2_index, batch_size = 10, num_workers = 8)
val_loader = torch.utils.data.DataLoader(trainset, sampler = val_index, batch_size = 10, num_workers = 8)
test_loader = torch.utils.data.DataLoader(testset, sampler = test_index)  #test set for running every epoch needs to be small
test_loader_big = torch.utils.data.DataLoader(testset)


# Not gonna lie not entirely sure what this does but I see people do this


set1dataiter = iter(set1_loader)
set1images, set1labels = set1dataiter.next()
set2dataiter = iter(set2_loader)
set2images, set2labels = set2dataiter.next()
valdataiter = iter(val_loader)
valimages, vallabels = valdataiter.next()



#CNN blocks for the backbone

class CNNBlock1(nn.Module):
    def __init__(self):
        super(CNNBlock1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
        torch.nn.init.xavier_normal_(self.conv1.weight)  #use xavier normal initialisation for all
        self.conv2 = nn.Conv2d(16, 48, kernel_size = 3)
        torch.nn.init.xavier_normal_(self.conv2.weight)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        return x
cnnblock1 = CNNBlock1()


class CNNBlock2(nn.Module):
    def __init__(self):
        super(CNNBlock2, self).__init__()
        self.conv3 = nn.Conv2d(48, 96, kernel_size = 3, padding = 1)
        torch.nn.init.xavier_normal_(self.conv3.weight)        
        self.conv4 = nn.Conv2d(96, 192, kernel_size = 4, padding = 1)
        torch.nn.init.xavier_normal_(self.conv4.weight)
            
    def forward(self, x):
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x),2))
        return x
cnnblock2 = CNNBlock2()

class CNNBlock3(nn.Module):
    def __init__(self):
        super(CNNBlock3, self).__init__()
        self.conv5 = nn.Conv2d(192, 192, kernel_size = 3, padding = 1)
        torch.nn.init.xavier_normal_(self.conv5.weight)
        self.conv6 = nn.Conv2d(192, 384, kernel_size = 2)
        torch.nn.init.xavier_normal_(self.conv6.weight)
    
    def forward(self, x):
    	#This block has no maxpooling
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x
cnnblock3 = CNNBlock3()

#MLP block for the branch
class MLPBlock(nn.Module):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(384, 192)
        torch.nn.init.xavier_normal_(self.fc1.weight)        
        self.fc2 = nn.Linear(192, 48)
        torch.nn.init.xavier_normal_(self.fc2.weight) 
        self.fc3 = nn.Linear(48, 10)
        torch.nn.init.xavier_normal_(self.fc3.weight) 
    
    def forward(self, x):
        x = F.avg_pool2d(x,6)
        x = x.view(-1, 1*1*384)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
mlpblock = MLPBlock()

#Alternate Freeze Model

class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.cnnblock1 = cnnblock1
        self.cnnblock2 = cnnblock2
        self.cnnblock3 = cnnblock3
        self.mlpblock = mlpblock
    
    def forward(self, x):
        x1 = self.cnnblock1(x)
        x2 = self.cnnblock2(x1)
        x3 = self.cnnblock3(x2)
        x4 = self.mlpblock(x3)
        return F.log_softmax(x4, dim=1)

ensemblemodel = EnsembleModel()
ensemblemodel.to(device)

# In[9]:


# In[14]:


optimizer = optim.SGD(ensemblemodel.parameters(), lr = learnrate)


# In[15]:


criterion = nn.CrossEntropyLoss()


# In[10]:


trainingloss = []
validationloss = []
testaccuracy = []
traininglossnoisy = []
validationlossnoisy = []
for epoch in range(nepochs):

    if epoch % 2 == 0 and epoch > 125:
        for param in cnnblock1.parameters():
            param.requires_grad_(False)
        
        for param in cnnblock2.parameters():
            param.requires_grad_(False)
        
        for param in cnnblock3.parameters():
            param.requires_grad_(False)

        for param in mlpblock.parameters():
            param.requires_grad_(True)
        
        running_loss = 0.0


        for i, data in enumerate(set2_loader,0):
            inputs, set2labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = ensemblemodel(inputs)
            loss = criterion(outputs, set2labels)
            loss.backward()
            grads = ensemblemodel.mlpblock.fc3.weight.grad
            optimizer.step()
            
            #print stats
            running_loss += loss.item()
            if i%500 == 499:
                trainingloss.append(running_loss/500)
                traininglossnoisy.append(running_loss/500)
                print(grads)
                running_loss = 0.0
        running_loss2 = 0.0
        
        for i,data in enumerate(val_loader): 
            inputs,vallabels = data[0].to(device),data[1].to(device)
            outputs = ensemblemodel(inputs)
            lloss = criterion(outputs, vallabels)   	
                
            running_loss2 += lloss.item()
            if i%500 == 499:
                validationloss.append(running_loss2/500)
                validationlossnoisy.append(running_loss2/500)
                running_loss2 = 0.0
                
                
    elif epoch % 2 == 1 and epoch > 125:
        for param in cnnblock1.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock2.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock3.parameters():
            param.requires_grad_(True)
            
        for param in mlpblock.parameters():
            param.requires_grad_(False)
        
        running_loss = 0.0
        
        for i, data in enumerate(set1_loader,0):
            inputs, set1labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = ensemblemodel(inputs)
            loss = criterion(outputs, set1labels)
            loss.backward()
            grads = ensemblemodel.cnnblock3.conv6.weight.grad
            optimizer.step()
            
            #print stats
            running_loss += loss.item()
            if i%500 == 499:
                traininglossnoisy.append(running_loss/500)            	                
                print(grads)
                running_loss = 0.0

        running_loss2 = 0.0
        for i,data in enumerate(val_loader): 
            inputs,vallabels = data[0].to(device), data[1].to(device)
            outputs = ensemblemodel(inputs)
            lloss = criterion(outputs, vallabels)   	
                
            running_loss2 += lloss.item()
            if i%500 == 499:
                validationlossnoisy.append(running_loss2/500)
                running_loss2 = 0.0     
                
    else:
        for param in cnnblock1.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock2.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock3.parameters():
            param.requires_grad_(True)
            
        for param in mlpblock.parameters():
            param.requires_grad_(True)
        
        running_loss = 0.0
        
        for i, data in enumerate(set1_loader,0):
            inputs, set1labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = ensemblemodel(inputs)
            loss = criterion(outputs, set1labels)
            loss.backward()
            grads = ensemblemodel.mlpblock.fc3.weight.grad
            optimizer.step()
            
            #print stats
            running_loss += loss.item()
            if i%500 == 499:
                traininglossnoisy.append(running_loss/500)
                print(grads)            	                
                running_loss = 0.0

        running_loss2 = 0.0
        for i,data in enumerate(val_loader): 
            inputs,vallabels = data[0].to(device), data[1].to(device)
            outputs = ensemblemodel(inputs)
            lloss = criterion(outputs, vallabels)   	
                
            running_loss2 += lloss.item()
            if i%500 == 499:
                validationlossnoisy.append(running_loss2/500)
                running_loss2 = 0.0                                 
                
    correct_count, all_count = 0, 0
    for i, data in enumerate(test_loader,0):
      inp,labels = data[0].to(device),data[1].to(device)
      with torch.no_grad():
            logps = ensemblemodel(inp)

      ps = torch.exp(logps)
      ps = ps.cpu()
      probab = list(ps.numpy()[0])
      pred_label = probab.index(max(probab))
      true_label = labels.cpu()
      if(true_label == pred_label):
          correct_count += 1
      all_count += 1

    print("\nModel Accuracy =", (correct_count/all_count))
    testaccuracy.append(correct_count/all_count)
    print(epoch)	                
                
print('Finished Training')

torch.save(ensemblemodel.state_dict(), '/home/brian_chen/mytitandir/ensemblemodel.pth')

class CNNBlock12(nn.Module):
    def __init__(self):
        super(CNNBlock12, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(16, 48, kernel_size = 3)
        torch.nn.init.xavier_normal_(self.conv2.weight)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        return x
cnnblock12= CNNBlock12()


# In[3]:


class CNNBlock22(nn.Module):
    def __init__(self):
        super(CNNBlock22, self).__init__()
        self.conv3 = nn.Conv2d(48, 96, kernel_size = 3, padding = 1)
        torch.nn.init.xavier_normal_(self.conv3.weight)        
        self.conv4 = nn.Conv2d(96, 192, kernel_size = 4, padding = 1)
        torch.nn.init.xavier_normal_(self.conv4.weight)
            
    def forward(self, x):
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x),2))
        return x
cnnblock22 = CNNBlock22()

# In[4]:


class CNNBlock32(nn.Module):
    def __init__(self):
        super(CNNBlock32, self).__init__()
        self.conv5 = nn.Conv2d(192, 192, kernel_size = 3, padding = 1)
        torch.nn.init.xavier_normal_(self.conv5.weight)
        self.conv6 = nn.Conv2d(192, 384, kernel_size = 2)
        torch.nn.init.xavier_normal_(self.conv6.weight)
    
    def forward(self, x):
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x
cnnblock32 = CNNBlock32()



# In[5]:
class MLPBlock2(nn.Module):
    def __init__(self):
        super(MLPBlock2, self).__init__()
        self.fc1 = nn.Linear(384, 192)
        torch.nn.init.xavier_normal_(self.fc1.weight) 
        self.fc2 = nn.Linear(192, 48)
        torch.nn.init.xavier_normal_(self.fc2.weight) 
        self.fc3 = nn.Linear(48, 10)
        torch.nn.init.xavier_normal_(self.fc3.weight) 
    
    def forward(self, x):
        x = F.avg_pool2d(x,6)
        x = x.view(-1, 1*1*384)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
mlpblock2 = MLPBlock2()


class EnsembleModel2(nn.Module):
    def __init__(self):
        super(EnsembleModel2, self).__init__()
        self.cnnblock12 = cnnblock12
        self.cnnblock22 = cnnblock22
        self.cnnblock32 = cnnblock32
        self.mlpblock2 = mlpblock2
    
    def forward(self, x):
        x1 = self.cnnblock12(x)
        x2 = self.cnnblock22(x1)
        x3 = self.cnnblock32(x2)
        x4 = self.mlpblock2(x3)
        return F.log_softmax(x4, dim=1)

ensemblemodel2 = EnsembleModel2()
ensemblemodel2.to(device)

optimizer2 = optim.SGD(ensemblemodel2.parameters(), lr = learnrate)

trainingloss2 = []
validationloss2 = []
testaccuracy2 = []
traininglossnoisy2 = []
validationlossnoisy2 = []
for epoch in range(nepochs):

    if epoch % 2 == 0:
        for param in cnnblock12.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock22.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock32.parameters():
            param.requires_grad_(True)
        
        for param in mlpblock2.parameters():
            param.requires_grad_(True)
        
        running_loss = 0.0
        for i, data in enumerate(set2_loader,0):
            inputs, set2labels = data[0].to(device), data[1].to(device)
            
            optimizer2.zero_grad()
            
            outputs = ensemblemodel2(inputs)
            loss = criterion(outputs, set2labels)
            loss.backward()
            grads = ensemblemodel2.mlpblock2.fc3.weight.grad
            optimizer2.step()
            
            #print stats
            running_loss += loss.item()
            if i%500 == 499:
                trainingloss2.append(running_loss/500)
                traininglossnoisy2.append(running_loss/500)
                running_loss = 0.0
                print(grads)

        running_loss2 = 0.0
        
        for i,data in enumerate(val_loader): 
            inputs,vallabels = data[0].to(device), data[1].to(device)
            outputs = ensemblemodel2(inputs)
            lloss = criterion(outputs, vallabels)   	
                
            running_loss2 += lloss.item()
            if i%500 == 499:
                validationloss2.append(running_loss2/500)
                validationlossnoisy2.append(running_loss2/500)
                running_loss2 = 0.0                
  
                
    else:
        for param in cnnblock12.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock22.parameters():
            param.requires_grad_(True)
        
        for param in cnnblock32.parameters():
            param.requires_grad_(True)
        
        for param in mlpblock2.parameters():
            param.requires_grad_(True)
            
        
        running_loss = 0.0
        
        for i, data in enumerate(set1_loader,0):
            inputs, set1labels = data[0].to(device), data[1].to(device)	
            
            optimizer2.zero_grad()
            
            outputs = ensemblemodel2(inputs)
            loss = criterion(outputs, set1labels)
            loss.backward()
            grads = ensemblemodel2.mlpblock2.fc3.weight.grad
            optimizer2.step()
            
            #print stats
            running_loss += loss.item()
            if i%500 == 499:
                traininglossnoisy2.append(running_loss/500)	                
                print(grads)
                running_loss = 0.0
        
        for i,data in enumerate(val_loader): 
            inputs,vallabels = data[0].to(device), data[1].to(device)
            outputs = ensemblemodel2(inputs)
            lloss = criterion(outputs, vallabels)   	
                
            running_loss2 += lloss.item()
            if i%500 == 499:
                validationlossnoisy2.append(running_loss2/500)
                running_loss2 = 0.0     
                
                
    correct_count, all_count = 0, 0
    for i,data in enumerate(test_loader,0):
      inp,labels = data[0].to(device),data[1].to(device)
      with torch.no_grad():
      	logps = ensemblemodel2(inp)
    
      ps = torch.exp(logps)
      ps = ps.cpu()
      probab = list(ps.numpy()[0])
      pred_label = probab.index(max(probab))
      true_label = labels.cpu()
      if(true_label == pred_label):
      	correct_count += 1
      all_count += 1

    print("\nModel Accuracy =", (correct_count/all_count))
    testaccuracy2.append(correct_count/all_count)
    print(epoch)
                
                
print('Finished Training')
torch.save(ensemblemodel2.state_dict(), '/home/brian_chen/mytitandir/ensemblemodel2.pth')

fig1, ax1 =plt.subplots()
ax1.plot(trainingloss)
ax1.plot(trainingloss2)
ax1.legend(["Alternate Freeze", "Control"])
ax1.set_xlabel("2 epochs", fontsize = 20)
ax1.set_ylabel("training loss", fontsize = 20)
anchored_text = AnchoredText('set1:{} set2:{} lr:{} \n optim:{} activ:{}'.format(train1,train2,learnrate,OPTIM,activation),loc='upper left', prop = dict(fontweight = "normal", size= 10))
ax1.add_artist(anchored_text)
plt.savefig('single_training_loss35')


plt.clf()

fig2, ax2 =plt.subplots()
ax2.plot(traininglossnoisy)
ax2.plot(traininglossnoisy2)
ax2.legend(["Alternate Freeze", "Control"])
ax2.set_xlabel("epochs", fontsize = 20)
ax2.set_ylabel("training loss", fontsize = 20)
anchored_text = AnchoredText('set1:{} set2:{} lr:{} \n optim:{} activ:{}'.format(train1,train2,learnrate,OPTIM,activation),loc='upper left', prop = dict(fontweight = "normal", size= 10))
ax2.add_artist(anchored_text)
plt.savefig('single_training_noisy_loss35')

plt.clf()

fig3, ax3 =plt.subplots()
ax3.plot(testaccuracy)
ax3.plot(testaccuracy2)
ax3.legend(["Alternate Freeze", "Control"])
ax3.set_xlabel("epochs", fontsize = 20)
ax3.set_ylabel("test accuracy", fontsize = 20)
anchored_text = AnchoredText('set1:{} set2:{} lr:{} \n optim:{} activ:{}'.format(train1,train2,learnrate,OPTIM,activation),loc='upper left', prop = dict(fontweight = "normal", size= 10))
ax3.add_artist(anchored_text)
plt.savefig('single_test_accuracy35')

plt.clf()
fig4, ax4 =plt.subplots()
ax4.plot(validationloss)
ax4.plot(validationloss2)
ax4.legend(["Alternate Freeze", "Control"])
ax4.set_xlabel("2 epochs", fontsize = 20)
ax4.set_ylabel("validation loss", fontsize = 20)
anchored_text = AnchoredText('set1:{} set2:{} val:{}\n lr:{} optim:{} activ:{}'.format(train1,train2,valsiz,learnrate,OPTIM,activation),loc='upper left', prop = dict(fontweight = "normal", size= 10))
ax4.add_artist(anchored_text)
plt.savefig('single_validation_loss35') 

plt.clf()

fig5, ax5 =plt.subplots()
ax5.plot(validationlossnoisy)
ax5.plot(validationlossnoisy2)
ax5.legend(["Alternate Freeze", "Control"])
ax5.set_xlabel("epochs", fontsize = 20)
ax5.set_ylabel("validation loss", fontsize = 20)
anchored_text = AnchoredText('set1:{} set2:{} val:{}\n lr:{} optim:{} activ:{}'.format(train1,train2,valsiz,learnrate,OPTIM,activation),loc='upper left', prop = dict(fontweight = "normal", size= 10))
ax5.add_artist(anchored_text)
plt.savefig('single_validation_noisy_loss35')


