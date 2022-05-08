import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

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
        x = torch.hstack((state,t))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x  #(p,q)

#model = torch.load('Weak_HO_30k.pt', map_location=torch.device('cpu')) #Without semigroup loss
model = torch.load('semigroup_HO.pt', map_location=torch.device('cpu')) #Without semigroup loss

#Initial conditions
p0,q0 = 0,1

#Neural Net
model.eval()
with torch.no_grad(): #Tell torch to stop keeping track of gradients
    t = torch.linspace(0, 10, 50, dtype=torch.float)
    dt = torch.reshape(t[1]-t[0],(1,1))
    S = torch.tensor([[p0,q0]])
    p = torch.zeros(len(t))
    q = torch.zeros(len(t))
    p[0],q[0] = p0,q0
    for i in range(1,len(t)):
        S = model(S,dt)
        p[i] = S[0,0]
        q[i] = S[0,1]

    plt.plot(t, p, label="Neural Net: p")
    plt.plot(t, q, label="Neural Net: q")

#Finite difference
t = torch.linspace(0, 10, 10000, dtype=torch.float)
p,q = harmonic_oscillator(p0,q0,t)
plt.plot(t, p, label="Finite Diff: p")
plt.plot(t, q, label="Finite Diff: q")

plt.legend()
plt.savefig('Semigroup_test.png')
plt.show()






