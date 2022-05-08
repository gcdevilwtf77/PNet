import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, n_hidden=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,n_hidden)
        self.fc4 = nn.Linear(n_hidden,2)

    def forward(self, x):
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
optimizer = optim.SGD(model.parameters(), lr=0.05)  #Learning rate
scheduler = StepLR(optimizer, step_size=1, gamma=0.99995)

#Batch size
batch_size = 1000
epochs = 100000

#Step size
h = 0.001

model.train()
#Training epochs
for i in range(epochs):


    x = 10*torch.rand((batch_size,1), dtype=torch.float).to(device)
    x[0] = 0
    
    #Construct loss function and compute gradients and optimizer updates
    optimizer.zero_grad()
    yt = (model(x+h) - model(x-h))/(2*h)
    pt,qt = yt[:,0],yt[:,1]
    y = model(x)
    p,q = y[:,0],y[:,1]

    loss = torch.mean((qt - p)**2 + (pt + q)**2) + p[0]**2 + (q[0]-1)**2
    loss.backward()
    print(i,loss.item())

    optimizer.step()
    scheduler.step()





device = torch.device('cpu')
model.to(device)
model.eval()
with torch.no_grad(): #Tell torch to stop keeping track of gradients
    x = torch.linspace(0, 10, batch_size).to(device)[:,None]
    plt.figure()
    y = model(x)
    p,q = y[:,0],y[:,1]
    plt.plot(x, p, label="p")
    plt.plot(x, q, label="q")
    plt.legend()
    plt.savefig('DeepLearning_pt_qt_1e5epochs_batch10000.png')

    plt.figure()
    plt.plot(p,q)
    plt.xlabel('p')
    plt.ylabel('q')
    plt.savefig('DeepLearning_p_vs_q_1e5epochs_batch10000.png')

















