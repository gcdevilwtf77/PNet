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
        self.fc1 = nn.Linear(2,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,n_hidden)
        self.fc4 = nn.Linear(n_hidden,2)

    def forward(self, state):
        x = state
        # x = torch.hstack((state,t))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x  #(p,q)

def F_HO(x):
    """Dynamics for harmonic oscillator"""

    p = x[:,0]
    q = x[:,1]
    Fs = torch.vstack((-q,p)).T
    return Fs

Model = torch.load('EulerCorrection.pt', map_location=torch.device('cpu'))

#Initial conditions
p0,q0 = 0,1

Model.eval()
plt.figure()
with torch.no_grad(): #Tell torch to stop keeping track of gradients
    t = torch.linspace(0, 100, 100, dtype=torch.float)
    dt = torch.reshape(t[1]-t[0],(1,1))
    S = torch.tensor([[p0,q0]])
    p = torch.zeros(len(t))
    q = torch.zeros(len(t))
    h = 0.01
    p[0],q[0] = p0,q0
    for i in range(1,len(t)):
        S = S + h*F_HQ(S) + h**2/2*Model(S)
        p[i] = S[0,0]
        q[i] = S[0,1]

    plt.plot(t, p, label="Neural Net: p")
    plt.plot(t, q, label="Neural Net: q")
    plt.legend()
    plt.show()