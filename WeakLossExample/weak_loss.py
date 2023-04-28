"""Weak Loss for Solving ODEs with Neural Networks"""
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import torch
import utils


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

#Rule
#rule = 'midpoint' 
#rule = 'simpson' 
rule = 'trapezoid' 
#rule = 'rightpoint'
#rule = 'leftpoint'

#Set up device and model
cuda = True
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net(100).to(device)

#Initialization of variables
T = np.pi #final time
epochs = 10000
batch_size = 1000
dx = T/batch_size
x = torch.arange(dx/2,T,dx).reshape((batch_size,1)).to(device)

#Set up optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.01)  #Learning rate
scheduler = StepLR(optimizer, step_size=1, gamma=0.001**(1/epochs))

#Training epochs
model.train()
for i in range(epochs):

    #Construct loss function and compute gradients and optimizer updates
    #We are solving the ODE y'(x) = -y(x) + cos(x) + sin(x), y(0)=0
    #whose solution is y(x)=sin(x). We're learning a function 
    #y(x)=y(0) + y'(0)x + (1/2)f(x)x^2,
    #where f(x) is the neural network.

    optimizer.zero_grad()
    loss = dx*torch.sum(torch.abs(utils.y(model,x+dx/2) - utils.integrate(model,x,dx,rule))) 
    loss.backward()
    optimizer.step()
    scheduler.step()

    if i % 1000 == 0:
        print(i,loss.item())

torch.save(model,'weak_loss_model.pt')


