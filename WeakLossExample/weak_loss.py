"""Weak Loss for Solving ODEs with Neural Networks"""
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import torch

#Our neural network as 2 layers with 100 hidden nodes 
class Net(nn.Module):
    def __init__(self, n_hidden=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,1)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Use double precision everywhere
torch.set_default_dtype(torch.float64)

#Set up device and model
cuda = True
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net(100).to(device)

#Initialization of variables
T = np.pi #final time
batch_size = 1000
dx = T/batch_size
epochs = 10000
x = torch.arange(dx,T+dx,dx).reshape((batch_size,1)).to(device)
phi = 1 - torch.exp(-x)
x_mid = x - dx/2
phi_mid = 1 - torch.exp(-x_mid)

#Set up optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.01)  #Learning rate
scheduler = StepLR(optimizer, step_size=1, gamma=0.01**(1/epochs))

#Training epochs
model.train()
for i in range(epochs):

    #Construct loss function and compute gradients and optimizer updates
    #We are solving the ODE y'(x) = y(x) + cos(x) - sin(x), y(0)=0
    #whose solution is y(x)=sin(x). We're learning a function y(x)=phi(x)f(x)
    #where f(x) is the neural network and phi encodes the initial condition
    #so phi(0)=0. We chose phi(x) = 1 - e^{-x}.
    
    optimizer.zero_grad()
    y_mid = phi_mid*model(x_mid)
    y = phi*model(x)
    F = y_mid + torch.cos(x_mid) - torch.sin(x_mid)
    loss = dx*torch.sum(torch.abs(y - dx*torch.cumsum(F,dim=0))) #L1 loss
    loss.backward()

    optimizer.step()
    scheduler.step()

    if i % 1000 == 0:
        print(i,loss.item())

torch.save(model,'weak_loss_model.pt')


