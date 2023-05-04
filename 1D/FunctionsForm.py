import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
cwd = os.getcwd() ##windows specific
import matplotlib.pyplot as plt
# from LossRules import 
# threading.stack_size(200000000)
import warnings
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
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
    t = torch.linspace(0, b, n+1)[:,None]
    h = t[1]-t[0]
    t = t[:-1] + h/2
    t = t.to(device)

    #Evaluate the model
    ones = torch.ones((n,1)).to(device)
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
    print(rule)

except:
    batch_size = 1000
    epochs = int(1e5)
    rule = 'simpson_rule'

print(rule,batch_size,epochs)

#Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)  #Learning rate
scheduler = StepLR(optimizer, step_size=1, gamma= 0.01**(1/epochs))
# gamma=0.001**(1/epochs)



model.train()
#Training epochs
x = torch.ones(2).to(device)
x[0] = 0
x[1] = -1
x = torch.reshape(x, (1, 2))
f_T = torch.ones(1).to(device)*0.1
Results = torch.empty((int(1e4),6))
j = 0
k = 0
losses = []
for i in range(epochs):

    #Random initial conditions for p0 and q0
    # x = (2*torch.rand(2)-1).to(device)
    # x = torch.reshape(x,(1,2))

    #Random final time
    # T = 0.2*torch.rand(1).to(device)
    # T = max(T,torch.ones(1).to(device)*0.1)
    # T = torch.linspace(0,1,500)

    if i > 1000:
        if losses[-1] < 10**(-3):
            if losses[-1] < losses[-2]:
                f_T += 10**(-4)

    #Weak formulation loss
    optimizer.zero_grad()
    loss_weak = 0
    for z in range(10):
        T = f_T*torch.rand(1).to(device)
        model_value = model(x,T)
        loss_weak += torch.sum((model(x,T) - x - integrate(model,x,T,batch_size,device,rule,F=F_HO))**2)
    loss_weak = loss_weak/(z+1)

    #Semigroup loss
    # s1 = torch.rand((1,1)).to(device)
    # s2 = (1-s1)*torch.rand((1,1)).to(device)
    # loss_semigroup = torch.sum((model(x,s1+s2) - model(model(x,s1),s2))**2)
    # loss_semigroup += torch.sum((model(x,s1+s2) - model(model(x,s2),s1))**2)

    #Full loss
    loss = loss_weak #+ loss_semigroup
    losses.append(loss)
    #Gradient descent
    loss.backward()

    Results[j,0:2] = x
    Results[j,2:3] = T
    Results[j,3:5] = model_value
    Results[j,5:6] = loss
    j += 1
    if i == 0  or (i+1)%(int(epochs/10**2)) ==0:
        print(i,loss.item(),f_T,x)

    if (i+1)%2 == 0:
        if k == 0:
            mode_results = 'w'
            k+=1
        else:
            mode_results = 'a'
        df = pd.DataFrame(Results.detach().numpy(), columns=['p_0', 'q_0', 'T',
                                                             'p_T', 'q_T', 'loss'])
        df.to_csv(cwd+'/Results/model_results_rule_' + rule + '_batch_size_' +
                  str(batch_size) + '_epochs_' + str(epochs) + '.csv', index=False,
                  mode= mode_results)
        Results = torch.empty((int(1e4), 6))
        j = 0
    optimizer.step()
    scheduler.step()

torch.save(model,'semigroup_Function_rule_' + rule + '_batch_size_' + str(batch_size) +
           '_epochs_' + str(epochs) +'.pt')

# device = torch.device('cpu')
# model.to(device)
# t = torch.linspace(0, 1, batch_size)[:,None].to(device)
# ones = torch.ones((batch_size,1)).to(device)
# model.eval()
# with torch.no_grad(): #Tell torch to stop keeping track of gradients
#     for i in range(10):
#         plt.figure()
#         x = (2*torch.rand(2)-1).to(device)
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







