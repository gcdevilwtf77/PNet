import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
# from LossRules import

def true_solution(x,T):
    # if x.size()[0] > 1:
    #     print(x)
    return torch.hstack((x[0][0]*torch.cos(T) - x[0][1]*torch.sin(T),
                      x[0][0]*torch.sin(T) + x[0][1]*torch.cos(T)))

def F_HO(x):
    """Dynamics for harmonic oscillator"""

    p = x[:,0]
    q = x[:,1]
    F = torch.vstack((-q,p)).T
    return F

def integrate(model,x,b,n,device,rule,F=None):
    """General function for integration of F(model(x,t)) from t=0 to t=b
    using n points and midpoint rule. F is the dynamics.
    """

    #Convert a,b to float
    if type(b) == torch.Tensor:
        b = b.cpu().numpy()[0]

    #Set up points
    t = torch.linspace(0, b, n+1, dtype=torch.float)[:,None]
    h = t[1]-t[0]
    t = t[:-1] + h/2
    t = t.to(device)

    #Evaluate the model
    ones = torch.ones((n,1), dtype=torch.float).to(device)
    if F is None:
        S = model(ones*x,t)
    else:
        S = F(model(ones*x,t))

    #Integrate
    h = torch.tensor(h).to(device)

    if rule == 'left_point_rule':
        left_point_rule = h*torch.sum(S[:-1,:],dim=0)
        return left_point_rule

    elif rule == 'right_point_rule':
        right_point_rule = h*torch.sum(S[1:,:],dim=0)
        return right_point_rule

    elif rule == 'mid_point_rule':
        mid_point_rule = h*torch.sum(S,dim=0)
        return mid_point_rule

    elif rule == 'trapezoid_rule':
        trapezoid_rule = (h/2)*(S[0,:] + 2*torch.sum(S[1:-1,:],dim=0) + S[-1,:])
        return trapezoid_rule

    elif rule == 'simpson_rule':
        simpson_rule = (h/3)*(S[0,:] + 4*torch.sum(S[:-1:2,:],dim=0) + 2*torch.sum(S[1:-1:2,:],dim=0) + S[-1,:])
        return simpson_rule

def harmonic_oscillator(p0,q0,t):
    """Finite difference solver for harmonic oscillator"""

    n = len(t)

    p = np.zeros(n)
    q = np.zeros(n)
    p[0],q[0] = p0,q0

    for i in range(1,n):
        h = (t[i]-t[i-1])
        p[i] = p[i-1] - h*q[i-1]
        q[i] = q[i-1] + h*p[i-1]

    return p,q

class Net(nn.Module):
    def __init__(self, n_hidden=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,n_hidden)
        self.fc4 = nn.Linear(n_hidden,2)

    def forward(self, state, t):
        if t.dim()==1:
            t = torch.reshape(t,(len(t),1))
        x = torch.hstack((state,t))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x  #(p,q)

if torch.cuda.is_available() == True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = Net(100).to(device)

#Batch size
rule_dic = {'0': 'left_point_rule', '1': 'right_point_rule', '2': 'mid_point_rule',
            '3': 'trapezoid_rule', '4': 'simpson_rule'}

try:
    rule_chosen = input("What rule would you like to use? Vaild choices are: 0:left_point_rule, " +
                         "1:right_point_rule, 2:mid_point_rule, 3:trapezoid_rule or 4:simpson_rule: ")
    batch_size_chosen = int(float(input("What batch size would you like? It must be nonnegative integer? ")))
    epochs_chosen  = int(float(input("How many epochs would you like? It must be nonnegative integer? ")))
    rule = rule_dic[rule_chosen]
    batch_size = batch_size_chosen
    epochs = epochs_chosen

except:
    batch_size = 1000
    epochs = int(1e5)
    rule = 'simpson_rule'


#Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)  #Learning rate
scheduler = StepLR(optimizer, step_size=1, gamma= 0.01**(1/epochs))
# gamma=0.001**(1/epochs)



model.train()
#Training epochs
for i in range(epochs):

    #Random initial conditions for p0 and q0
    x = (2*torch.rand(2, dtype=torch.float)-1).to(device)
    x = torch.reshape(x,(1,2))

    #Random final time
    T = 0.2*torch.rand(1, dtype=torch.float).to(device)
    T = max(T,torch.ones(1).to(device)*0.1)
    # T = torch.linspace(0,1,500)



    #Weak formulation loss
    # optimizer.zero_grad()
    # loss_weak = torch.sum((model(x,T) - x - integrate(model,x,T,batch_size,device,rule,F=F_HO))**2)
    #
    # #Exact solution
    # #p = c_1 cos(t) - c_2 sin(t)
    # #q = c_1 sin(t) + c_2 cos(t)

    rule = 'left_point_rule'
    batch_size = 1000
    epochs = int(1e5)
    exact_difference = true_solution(x,T) - x\
                     - integrate(true_solution,x,T,batch_size,device,rule,F=F_HO)

    if (abs(exact_difference)>10**(-4)).any():

        print(i,exact_difference)









