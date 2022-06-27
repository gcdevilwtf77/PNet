import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
# from LossRules import 

def F_HO(x):
    """Dynamics for harmonic oscillator"""

    p = x[:,0]
    q = x[:,1]
    F = torch.vstack((-q,p)).T
    return F

def NumericalCorrector(model,x,b,device,h,F):

    #Convert a,b to float
    if type(b) == torch.Tensor:
        b = b.cpu().numpy()[0]

    #Set up points
    Size = int(np.round(b,3)/h)
    if Size == 0:
        Size = 1
    S_h = torch.zeros((Size,2)).to(device)
    # t = torch.linspace(0, b, n+1, dtype=torch.float)[:,None]
    # h = t[1]-t[0]
    # t = t[:-1] + h/2
    # t = t.to(device)

    #Evaluate the model
    # ones = torch.ones((n,1), dtype=torch.float).to(device)
    for i in range(len(S_h)):
        if i == 0:
            # S_h[0,:] = x + h*F(x) + h**2/2*F(model(x))
            y = x + h*F(x) + h**2/2*F(model(x))
        else:
            # S_h[i,:] = S_h[i-1:i,:] + h*F(S_h[i-1:i,:]) + h**2/2*F(model(S_h[i-1:i,:]))
            y = y + h*F(y) + h**2/2*F(model(y))
        S_h[i,:] = y

    return S_h

def integrate(model,x,b,n,device,rule,F):
    """General function for integration of F(model(x,t)) from t=0 to t=b
    using n points and midpoint rule. F is the dynamics.
    """


    h = 0.001
    h = torch.tensor(h).to(device)

    S = NumericalCorrector(model,x,b,device,h,F)

    #Integrate

    if rule == 'left_point_rule':
        weak_rule = h*torch.sum(S[:-1,:],dim=0)

    elif rule == 'right_point_rule':
        weak_rule = h*torch.sum(S[1:,:],dim=0)

    elif rule == 'mid_point_rule':
        weak_rule = h*torch.sum(S,dim=0)

    elif rule == 'trapezoid_rule':
        weak_rule = (h/2)*(S[0,:] + 2*torch.sum(S[1:-1,:],dim=0) + S[-1,:])

    elif rule == 'simpson_rule':
        weak_rule = (h/3)*(S[0,:] + 4*torch.sum(S[:-1:2,:],dim=0) + 2*torch.sum(S[1:-1:2,:],dim=0) + S[-1,:])
        
    return S[-1,:] - S[0,:] - weak_rule

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
        self.fc1 = nn.Linear(2,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,n_hidden)
        self.fc4 = nn.Linear(n_hidden,2)

    def forward(self, state):
        x = state
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
batch_size = 1000
epochs = int(1e5)
rule = 'simpson_rule'

#Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)  #Learning rate
scheduler = StepLR(optimizer, step_size=1, gamma=1)
# gamma=0.001**(1/epochs)



model.train()
#Training epochs
for i in range(epochs):

    #Random initial conditions for p0 and q0
    x = (2*torch.rand(2, dtype=torch.float)-1).to(device)
    x = torch.reshape(x,(1,2))

    #Random final time
    T = torch.rand(1, dtype=torch.float).to(device)
    # T = torch.linspace(0,1,500)

    #Weak formulation loss
    optimizer.zero_grad()
    loss_weak = torch.sum(integrate(model,x,T,batch_size,device,rule,F=F_HO)**2)

    #Full loss
    loss = loss_weak

    #Gradient descent
    loss.backward()
    
    if i == 0  or (i+1)%1000==0:
            print(i,loss.item())

    optimizer.step()
    scheduler.step()

torch.save(model,'EulerCorrection.pt')

# device = torch.device('cpu')
# model.to(device)
# t = torch.linspace(0, 1, batch_size, dtype=torch.float)[:,None].to(device)
# ones = torch.ones((batch_size,1), dtype=torch.float).to(device)
# model.eval()
# with torch.no_grad(): #Tell torch to stop keeping track of gradientsprint
#     for i in range(10):
#         plt.figure()
#         x = (2*torch.rand(2, dtype=torch.float)-1).to(device)
#         y = model(ones*x,t)
#         p,q = y[:,0],y[:,1]
#         plt.plot(t, p, label="Neural Net: p")
#         plt.plot(t, q, label="Neural Net: q")

#         p,q = harmonic_oscillator(x[0],x[1],t)
#         plt.plot(t, p, label="Finite Diff: p")
#         plt.plot(t, q, label="Finite Diff: q")

#         plt.legend()
#         plt.savefig('Semigroup_HO_%d.png'%i)
#         plt.close()







