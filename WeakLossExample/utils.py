import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np

#Our neural network as 2 layers with 100 hidden nodes 
class Net(nn.Module):
    def __init__(self, n_hidden=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,1)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def y(model,x,y0):
    return y0[0] + y0[1]*x + (1/2)*model(x)*x**2

def integrate(model,F,x,dx,rule,y0):
    x_left = x - dx/2
    x_right = x + dx/2
    y_left, y_mid, y_right = y(model,x_left,y0), y(model,x,y0), y(model,x_right,y0)
    F_left, F_mid, F_right = F(x_left,y_left), F(x,y_mid), F(x_right,y_right)
    if rule == 'midpoint':
        return dx*torch.cumsum(F_mid, dim=0)
    if rule == 'leftpoint':
        return dx*torch.cumsum(F_left, dim=0)
    if rule == 'rightpoint':
        return dx*torch.cumsum(F_right, dim=0)
    if rule == 'simpson':
        return dx*torch.cumsum((F_left + 4*F_mid + F_right)/6, dim=0)
    if rule == 'trapezoid':
        return dx*torch.cumsum((F_left + F_right)/2, dim=0)

def train(F,y0,T,rule='trapezoid',epochs=10000,lr=0.01,batch_size=1000,cuda=True,num_hidden=100,savefile='model.pt'):

    torch.set_default_dtype(torch.float64)
    y0 = [y0.item(),F(torch.tensor(0),y0).item()]  #Augment initial condition with intial slope 

    #Set up device and model
    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(num_hidden).to(device)

    #Set up x
    dx = T/batch_size
    x = torch.arange(dx/2,T,dx).reshape((batch_size,1)).to(device)

    #Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)  #Learning rate
    scheduler = StepLR(optimizer, step_size=1, gamma=0.001**(1/epochs))

    model.train()
    for i in range(epochs):

        optimizer.zero_grad()
        loss = dx*torch.sum(torch.abs(y(model,x+dx/2,y0) - y0[0] - integrate(model,F,x,dx,rule,y0)))
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 1000 == 0:
            print(i, loss.item())

    torch.save(model, 'Models/'+savefile)

def plot(F,y0,T,y_true,batch_size=1000,model_name='weak_loss_model.pt',filename_prefix='F'):

    torch.set_default_dtype(torch.float64)
    y0 = [y0.item(),F(torch.tensor(0),y0).item()]  #Augment initial condition with intial slope 
    model = torch.load('Models/'+model_name, map_location=torch.device('cpu'))

    #Set up x
    dx = T/batch_size
    x = torch.arange(dx/2,T,dx).reshape((batch_size,1))

    model.eval()
    # Plot solution
    with torch.no_grad():  # Tell torch to stop keeping track of gradients
        f = model(x)
        net = y(model, x, y0).numpy()  
        true = y_true(x).numpy()
        x = x.numpy()

        plt.figure()
        plt.plot(x, net, label='Neural Net Solution')
        plt.plot(x, true, label='True Solution')
        plt.legend()
        plt.savefig('Figures/'+filename_prefix+'_NeuralNetPlot.pdf')

        plt.figure()
        plt.plot(x, np.absolute(net - true))
        plt.title('Error')
        plt.savefig('Figures/'+filename_prefix+'_NeuralNetErrorPlot.pdf')

        plt.figure()
        plt.plot(x, f.numpy(), label='Neural Net Corrector')
        plt.plot(x, 2 * (true - y0[0] - y0[1]*x) / x**2, label='True Corrector')
        plt.title('Neural net corrector')
        plt.legend()
        plt.savefig('Figures/'+filename_prefix+'_NeuralNetCorrector.pdf')

