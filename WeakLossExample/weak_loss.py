"""Weak Loss for Solving ODEs with Neural Networks"""
import numpy as np
import torch
import utils

def F1(x,y):
    return -y + torch.cos(x) + torch.sin(x) 
def y1(x):
    return torch.sin(x) #True solution for F1
def F2(x,y):
    return  -y + x 
def y2(x):
    return x - 1 + 2*torch.exp(-x) #True solution for F2
def F3(x,y):
    return -y + torch.exp(-x) 
def y3(x):
    return torch.cos(x) + torch.sin(x) + torch.exp(-x)/2 #True solution for F3
def F4(x,y):
    return -y**2 + torch.sin(x)**2 + torch.cos(x)
def y4(x):
    return torch.sin(x) #True solution for F4
def F5(x,y):
    return -y**2 + torch.cos(x)**2 - torch.sin(x)
def y5(x):
    return torch.cos(x) #True solution for F5
def F6(x,y):
    return 2*torch.sqrt(torch.relu(y))
def y6(x):
    return x**2 #True solution for F6

def F7(x,y):
    return 2*torch.sqrt(torch.relu(y)+1e-6)
def y7(x):
    return x**2 #True solution for F6

zero = torch.tensor(0)
odes = [(F1,y1,2,'F1',0.01,10000),(F2,y2,2,'F2',0.01,10000),(F3,y3,2,'F3',0.01,10000),(F4,y4,2,'F4',0.01,10000),
        (F5,y5,torch.pi,'F5',0.01,10000),
        (F6,y6,torch.pi,'F6',0.01,10000),(F6,y6,torch.pi,'F6_2',0.01,100000),
        (F6,y6,torch.pi,'F6_3',0.01,1000000),(F6,y6,torch.pi,'F6_4',0.01,10000000),
        (F7,y7,torch.pi,'F7',0.01,10000),(F7,y7,torch.pi,'F7_2',0.01,100000),
        (F7,y7,torch.pi,'F7_3',0.01,1000000), (F7,y7,torch.pi,'F7_4',0.01,10000000)] #List of (ODE,True solution,Final time,name)

#Loop over ODEs to train on
for (F,y,T,name,lr,epochs) in odes[-1:]:
    utils.train(F,y(zero),T,lr=lr,epochs=epochs,savefile=name+'_model.pt')
    utils.plot(F,y(zero),T,y,model_name=name+'_model.pt',filename_prefix=name)

