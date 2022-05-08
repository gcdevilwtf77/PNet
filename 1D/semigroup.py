import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

#Our neural network as 2 layers with 100 hidden nodes 
class Net(nn.Module):
    def __init__(self, n_hidden=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,n_hidden)
        self.fc4 = nn.Linear(n_hidden,1)

    def forward(self, x, t):
        x = torch.hstack((x,t))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

if torch.cuda.is_available() == True:
    device = torch.device('cuda')
else: 
    device = torch.device('cpu')
model = Net(100).to(device)

#Set up optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)  #Learning rate
scheduler = StepLR(optimizer, step_size=1, gamma=1)

#Batch size
batch_size = 1000
epochs = 30000

#Step size
t = torch.linspace(0, 1, batch_size+1, dtype=torch.float)[:,None]
h = t[1]-t[0]
t = t[:-1] + h/2
t = t.to(device)

#Ones
ones = torch.ones((batch_size,1), dtype=torch.float).to(device)

model.train()
#Training epochs
for i in range(epochs):


    #Random initial condition
    x = torch.rand((1,1), dtype=torch.float).to(device) 

    #Weak formulation loss
    optimizer.zero_grad()
    S = model(x*ones,t)
    loss_weak = (S[-1] - x + torch.mean(S))**2

    #Semigroup loss
    s1 = torch.rand((1,1), dtype=torch.float).to(device) 
    s2 = (1-s1)*torch.rand((1,1), dtype=torch.float).to(device) 
    loss_semigroup = (model(x,s1+s2) - model(model(x,s1),s2))**2
    loss_semigroup += (model(x,s1+s2) - model(model(x,s2),s1))**2

    #Full loss
    loss = loss_weak + loss_semigroup

    #Gradient descent
    loss.backward()
    print(i,loss.item())

    optimizer.step()
    scheduler.step()


torch.save(model,'semigroup30k.pt')

device = torch.device('cpu')
model.to(device)
t = t.to(device)
ones = ones.to(device)
model.eval()
#Plot the classification decision boundary
with torch.no_grad(): #Tell torch to stop keeping track of gradients
    for i in range(10):
        plt.figure()
        x = torch.rand((1,1), dtype=torch.float).to(device) 
        y = model(x*ones,t)
        plt.plot(t, y, label="Neural Net")
        plt.plot(t, x*torch.exp(-t), label="Exact")
        plt.legend()
        plt.savefig('Semigroup30k_%d.png'%i)
        plt.close()















