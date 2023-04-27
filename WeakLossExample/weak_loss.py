"""Weak Loss for Solving ODEs with Neural Networks"""
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import torch


def y(x):
    return x + (1/2)*model(x)*x**2
def F(x,y):
    return -y + torch.cos(x) + torch.sin(x)

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
rule = 'midpoint' 
#rule = 'simpson' 

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
x_right = torch.arange(dx,T+dx,dx).reshape((batch_size,1)).to(device)
x_mid = x_right - dx/2
x_left = x_right - dx

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
    y_left, y_mid, y_right = y(x_left), y(x_mid), y(x_right)
    F_left, F_mid, F_right = F(x_left,y_left), F(x_mid,y_mid), F(x_right,y_right)
    if rule == 'midpoint':
        loss = dx*torch.sum(torch.abs(y_right - dx*torch.cumsum(F_mid, dim=0)))
        #loss = dx*torch.sum(torch.abs(y_right - dx*torch.cumsum(F_mid, dim=0))/x_mid**2) #I experimented with this weighting
    if rule == 'simpson':
        loss = dx*torch.sum(torch.abs(y_right - dx*torch.cumsum((F_left + 4*F_mid + F_right)/6, dim=0))) 
    loss.backward()

    optimizer.step()
    scheduler.step()

    if i % 1000 == 0:
        print(i,loss.item())

torch.save(model,'weak_loss_model.pt')


