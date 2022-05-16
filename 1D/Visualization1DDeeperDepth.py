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
        self.fc1 = nn.Linear(3,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,n_hidden)
        self.fc4 = nn.Linear(n_hidden,2)

    def forward(self, state, t):
        x = torch.hstack((state,t))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x  #(p,q)

SemiGroup2D = []
SemiGroup2D.append(torch.load('semigroup_2D.pt', map_location=torch.device('cpu'))) #semigroup loss

p10,p20,q10,q20 = 0,0,1,1
for j in range(len(SemiGroup2D)):
    SemiGroup2D[j].eval()
    plt.figure()
    with torch.no_grad(): #Tell torch to stop keeping track of gradients
        t = torch.linspace(0, 10, 50, dtype=torch.float)
        dt = torch.reshape(t[1]-t[0],(1,1))
        S = torch.tensor([[p10,q10]])
        p1 = torch.zeros(len(t))
        q1 = torch.zeros(len(t))
        p1[0],p2[0],q1[0],q2[0] = p10,p20,q10,q20
        for i in range(1,len(t)):
            S = SemiGroup2D[j](S,dt)
            p1[i] = S[0,0]
            p2[i] = S[0,1]
            q1[i] = S[0,2]
            q2[i] = S[0,3]

        plt.plot(t, p1, label="Neural Net: p1")
        plt.plot(t, q1, label="Neural Net: q1")
        plt.legend()
        plt.show()