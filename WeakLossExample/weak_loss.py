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
    return -y**2 + torch.sin(x)**2 + 2*torch.sin(x)*torch.cos(x)
def y4(x):
    return x #Put the true solution for F4 here

zero = torch.tensor(0)
odes = [(F1,y1,2,'F1'),(F2,y2,2,'F2'),(F3,y3,2,'F3')] #List of (ODE,True solution,Final time,name)

#Loop over ODEs to train on
for (F,y,T,name) in odes:
    utils.train(F,y(zero),T,savefile=name+'_model.pt')
    utils.plot(F,y(zero),T,y,model_name=name+'_model.pt',filename_prefix=name)

