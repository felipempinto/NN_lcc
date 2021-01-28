import os
from cv2 import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

REBUILD_DATA = False#True

path = os.path.join(os.path.dirname(__file__),'EuroSAT',"EuroSAT")
types = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path,i))]
types = {j:i for i,j in enumerate(types)}

def train_data():
    training_data = []
    for lb in types:
        p = os.path.join(path,lb)
        for i in tqdm(os.listdir(p)):
            im = cv2.imread(os.path.join(p,i))
            eye = np.eye(len(types))[types[lb]]
            training_data.append([np.array(im), eye])

    np.random.shuffle(training_data)
    np.save("training_data.npy", training_data)

if REBUILD_DATA:
    train_data()


training_data = np.load('training_data.npy',allow_pickle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)

        x = torch.randn(3,64,64).view(-1,3,64,64)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,10)
    
    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))

        # print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        
        return x
        
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1,self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=1)


t1 = time.time()
X = torch.Tensor([i[0] for i in training_data]).view(-1,3,64,64)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])
print(f"Time to read: {time.time()-t1}")

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

# BATCH_SIZE = 100
# EPOCHS = 10

# def train(net):
#     optimizer = optim.Adam(net.parameters(),lr=0.001)
#     loss_function = nn.MSELoss()
#     for epoch in range(EPOCHS):
#         for i in tqdm(range(0,len(train_X),BATCH_SIZE)):
#             batch_X = train_X[i:i+BATCH_SIZE].view(-1,3,64,64)
#             batch_y = train_y[i:i+BATCH_SIZE]

#             batch_X,batch_y = batch_X.to(device),batch_y.to(device)

#             net.zero_grad()
#             outputs = net(batch_X)
#             loss = loss_function(outputs,batch_y)
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch: {epoch}. Loss: {loss}")

# def test(net):
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for i in tqdm(range(len(test_X))):
#             real_class = torch.argmax(test_y[i]).to(device)
#             net_out = net(test_X[i].view(-1,3,64,64).to(device))[0]
#             predicted_class = torch.argmax(net_out)
#             if predicted_class == real_class:
#                 correct +=1
#             total +=1

#     print("Accuracy:", round(correct/total,3))

def fwd_pass(X,y,train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs,y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs,y)

    if train:
        loss.backward()
        optimizer.step()
    return acc,loss

def test(size=32):

    random_start = np.random.randint(len(test_X)-size)
    X, y = test_X[random_start:random_start+size],test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc,val_loss = fwd_pass(X.view(-1,3,64,64).to(device),y.to(device))
    return val_acc,val_loss

MODEL_NAME = f'model-{time.time()}'

net = Net().to(device)
optimizer = optim.Adam(net.parameters(),lr=0.001)
loss_function = nn.MSELoss()

print(MODEL_NAME)

def train():
    BATCH_SIZE = 100
    EPOCHS = 30
    with open('model.log','a') as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0,len(train_X),BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,3,64,64).to(device)
                batch_y = train_y[i:i+BATCH_SIZE].to(device)

                acc,loss = fwd_pass(batch_X,batch_y,train=True)
                if i % 50 == 0:
                    val_acc,val_loss = test(size=100)
                    f.write(f'{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n')

train()











