
# coding: utf-8

# In[1]:


import torch
import numpy as np
import pickle
import copy
#get_ipython().run_line_magic('matplotlib', 'inline')
#import matplotlib.pyplot as plt


# In[2]:


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_train_data():
    '''
    loads training data: 50,000 examples with 3072 features
    '''
    X_train = None
    Y_train = None
    for i in range(1, 6):
        pickleFile = unpickle('cifar-10-batches-py/data_batch_{}'.format(i))
        dataX = pickleFile[b'data']
        dataY = pickleFile[b'labels']
        if type(X_train) is np.ndarray:
            X_train = np.concatenate((X_train, dataX))
            Y_train = np.concatenate((Y_train, dataY))
        else:
            X_train = dataX
            Y_train = dataY

    Y_train = Y_train.reshape(Y_train.shape[0], 1)

    return X_train.T, Y_train.T

def load_test_data():
    '''
    loads testing data: 10,000 examples with 3072 features
    '''
    X_test = None
    Y_test = None
    pickleFile = unpickle('cifar-10-batches-py/test_batch')
    dataX = pickleFile[b'data']
    dataY = pickleFile[b'labels']
    if type(X_test) is np.ndarray:
        X_test = np.concatenate((X_test, dataX))
        Y_test = np.concatenate((Y_test, dataY))
    else:
        X_test = np.array(dataX)
        Y_test = np.array(dataY)

    Y_test = Y_test.reshape(Y_test.shape[0], 1)

    return X_test.T, Y_test.T

def train_test_split(X_train, Y_train):
    """
    Randomly splits data into 80% train and 20% validation data

    """
    msk = np.random.rand(Y_train.shape[1]) < 0.8

    X_Train = X_train[:,msk]  
    X_val = X_train[:,~msk]

    Y_Train = Y_train[:,msk]  
    Y_val = Y_train[:,~msk]

    return X_Train, Y_Train, X_val, Y_val

def get_batch(X, Y, batch_size):
    """
    Expected Functionality: 
    given the full training data (X, Y), return batches for each iteration of forward and backward prop.
    """
    n_batches = Y.shape[1]/batch_size
    idx = np.arange(Y.shape[1])

    np.random.shuffle(idx)
    mini = np.array_split(idx, n_batches)

    return mini
    pass


# In[3]:


X_train,Y_train = load_train_data()
X_test, Y_test = load_test_data()
X_Train, Y_Train, X_Val, Y_Val = train_test_split(X_train,Y_train)

#Normalize
minn = np.min(X_Train, axis=1,keepdims=True)
maxx = np.max(X_Train, axis=1,keepdims=True)
X_Train= (X_Train - minn)/(maxx-minn)

'''
minnv = np.min(X_Val, axis=1,keepdims=True)
maxxv = np.max(X_Val, axis=1,keepdims=True)
'''
#X_Val= (X_Val - minnv)/(maxxv-minnv)
X_Val= (X_Val - minn)/(maxx-minn)

'''
minn = np.min(X_test, axis=1,keepdims=True)
maxx = np.max(X_test, axis=1,keepdims=True)
'''
X_Test= (X_test - minn)/(maxx-minn)


# In[4]:


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5)
        #self.conv2 = nn.Conv2d
        self.conv2 = nn.Conv2d(64, 64, 5)
#         self.fc1 = nn.Linear(128 * 5 * 5, 120)
#         #self.fc2 = nn.Linear(120, 10)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
        self.fc1 = nn.Linear(64 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 10)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


net = Net().cuda()


# In[5]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


# In[6]:


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


# In[7]:


trainacc=[]
valacc=[]
traincost = []
valcost = []
num = 100

for epoch in range(num):  # loop over the dataset multiple times
     
    running_loss = 0.0
    running_loss_val = 0.0
    train_accuracy = 0.0
    val_accuracy = 0.0
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0
    print(epoch, flush=True)
    for i, idx in enumerate(get_batch(X_Train, Y_Train,100), 0):
        # get the inputs
        x_batch = [X_Train.T[index] for index in idx]
        y_batch = [Y_Train.T[index] for index in idx]
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)
        y_batch_onehot = get_one_hot(y_batch,10)
        x_batch = x_batch.reshape(x_batch.shape[0],3,32,32)
        input_tensor = torch.from_numpy(x_batch)
        label_tensor = torch.from_numpy(y_batch_onehot)
        inputs = Variable(input_tensor.cuda()).float()
        labels = Variable(label_tensor.cuda()).long()
        true_labels = torch.max(labels,1)[1]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100), flush=True)
            traincost.append(running_loss/100)
            running_loss = 0.0
            
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == true_labels.data.long()).sum()
    train_accuracy = 100 * correct_train / (total_train + 0.0001)
        
    for i, idx in enumerate(get_batch(X_Val, Y_Val,100), 0):
        # get the inputs
        
        x_batch_val = [X_Val.T[index] for index in idx]
        y_batch_val = [Y_Val.T[index] for index in idx]
        x_batch_val = np.asarray(x_batch_val)
        y_batch_val = np.asarray(y_batch_val)
        y_batch_onehot_val = get_one_hot(y_batch_val,10)
        x_batch_val = x_batch_val.reshape(x_batch_val.shape[0],3,32,32)
        input_tensor_val = torch.from_numpy(x_batch_val)
        label_tensor_val = torch.from_numpy(y_batch_onehot_val)
        inputs_val = Variable(input_tensor_val.cuda()).float()
        labels_val = Variable(label_tensor_val.cuda()).long()
        true_labels_val = torch.max(labels_val,1)[1]
        optimizer.zero_grad()
        outputs_val = net(inputs_val)
        loss_val = criterion(outputs_val, torch.max(labels_val, 1)[1])
#         loss.backward()
#         optimizer.step()

        running_loss_val += loss_val.data[0]
        print("Hello", i)
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss_val: %.3f' %
                  (epoch + 1, i + 1, running_loss_val / 100), flush=True)
            valcost.append(running_loss_val/100)
            running_loss_val = 0.0
        
        _, predicted_val = torch.max(outputs_val.data, 1)
        total_val += labels_val.size(0)
        correct_val += (predicted_val == true_labels_val.data.long()).sum()
    val_accuracy = 100 * correct_val / total_val
    trainacc.append(train_accuracy)
    valacc.append(val_accuracy)
    if(val_accuracy>=70.2):
        print("Target Val Accuracy reached. Breaking", flush=True)
        break
    
    
        
        
print('Finished Training', flush=True)
print('Training Accuracy ',trainacc, flush=True)
print('validation Accuracy',valacc, flush=True)


# In[12]:


correct = 0
total = 0
idx = get_batch(X_Test, Y_test,100)

for idx_list in idx:
    x_batch_test = [X_Test.T[index] for index in idx_list]
    y_batch_test = [Y_test.T[index] for index in idx_list]
    x_batch_test = np.asarray(x_batch_test)
    y_batch_test1 = np.asarray(y_batch_test)
    y_batch_onehot_test = get_one_hot(y_batch_test1,10)
    label_tensor_test = torch.from_numpy(y_batch_onehot_test)
    test_labels = Variable(label_tensor_test.cuda()).long()
    true_labels = torch.max(test_labels,1)[1]
    x_batch_test = x_batch_test.reshape(x_batch_test.shape[0],3,32,32)    
    input_tensor_test = torch.from_numpy(x_batch_test)
    images = Variable(input_tensor_test.cuda()).float()
    outputs = net(images)
    ##### loss = criterion(outputs, torch.max(labels, 1)[1])
    _, predicted = torch.max(outputs.data, 1)
   
    total += labels.size(0)
    correct += (predicted == true_labels.data.long()).sum()


# In[13]:


print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total), flush=True)


# In[10]:

'''
plt.figure()
plt.plot(trainacc, 'r', label ='trainaccuracy')
plt.plot(valacc, 'b', label = 'valaccuracy')
plt.legend()
plt.savefig("test.png")
'''

# In[11]:


x_t = []
x_v = []

for i in range(len(traincost)):
    x_t.append((i+1)*100)

for i in range(len(valcost)):
    x_v.append((i+1)*100)

'''    
plt.figure()
plt.plot(x_t,traincost, 'b', label = 'traincost')
plt.plot(x_v,valcost, 'r', label = 'valcost')
plt.legend()
plt.savefig("test1.png")
'''

