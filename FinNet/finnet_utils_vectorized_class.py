import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import autograd
import matplotlib.pyplot as plt
import numpy as np
# torch.set_default_tensor_type(torch.DoubleTensor)

#Our neural network as 2 layers with 100 hidden nodes 
class Net(nn.Module):
    def __init__(self, n_hidden=100,output_size=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden,output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.tanh(self.fc4(x))
        x = self.fc4(x)
        return x

class ode(object):
    def __init__(self,F,y0,T,rule='trapezoid',epochs=10000,lr=0.01,batch_size=1000,cuda=True,num_hidden=100,
                 numerical=False,second_derivate_expanison=True,plot_labels=['x']):
        self.F = F
        self.y0 = y0
        self.T = T
        self.rule = rule
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.cuda = cuda
        self.num_hidden = num_hidden
        self.output_size = np.size(y0.numpy())
        self.numerical = numerical
        self.second_derivate_expanison = second_derivate_expanison

        if plot_labels[0] == 'x':
            self.plot_labels = []
            for i in range(np.size(y0.numpy())):
                self.plot_labels.append(plot_labels[0] + '_' + str(i+1))
        else:
            self.plot_labels=plot_labels

        # Set up device and model
        self.use_cuda = cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # self.model = Net(self.num_hidden, self.output_size).to(self.device)

        # Set up x
        self.dx = self.T/self.batch_size
        self.step = (T - 0)/self.batch_size
        # self.x = torch.arange(self.dx/2, self.T, self.dx).reshape((self.batch_size, 1)).to(self.device)
        self.x = torch.arange(0, self.T+self.step,step=self.step).reshape((self.batch_size+1, 1)).to(self.device)

        # self.model = Net(n_hidden=self.x.size()[0], output_size=self.output_size).to(self.device)
        # self.model = Net(n_hidden=16, output_size=self.output_size).to(self.device)

        input_size = 1
        output_size = self.output_size
        k = 16
        self.model = nn.Sequential(nn.Linear(input_size, k),
                              nn.Tanh(),
                              nn.Linear(k, k),
                              nn.Tanh(),
                              nn.Linear(k, k),
                              nn.Tanh(),
                              nn.Linear(k, output_size),
                              # nn.Tanh(),
                              ).to(self.device)

        torch.set_default_dtype(torch.float64)

    def y(self, model,x,y0):
        if self.output_size == 1 and len(plot_labels) == 1:
            return y0[0] + y0[1] * x + (1 / 2) * model(x) * x ** 2
        else:
           return y0[:self.output_size] + y0[self.output_size:]*x + (1/2)*model(x)*x**2

    def train(self,savefile='model.pt'):


        if self.output_size == 1:
            y0 = [self.y0.item(), self.F(torch.tensor(0), self.y0).item()]  #Augment initial condition with intial slope
        else:
            y0 = self.y0.tolist() + self.F(torch.tensor(0), self.y0).tolist()

        model = self.model

        # if output_size > 1:
        y0 = torch.tensor(y0).to(self.device)

        #Set up optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=self.lr)  #Learning rate
        scheduler = StepLR(optimizer, step_size=1, gamma=0.001**(1/self.epochs))
        # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        # model.train()
        #for their loss function
        loss_mse = nn.MSELoss()
        # loss_mse = nn.L1Loss()
        for i in range(self.epochs):

            optimizer.zero_grad()
            x_grad = self.x+self.dx/2
            # x_grad.requires_grad_()
            if self.second_derivate_expanison == True:
                y_output = self.y(model,self.x,y0)
                loss = 0
            else:
                y_output = model(self.x)
                loss = (y_output[0] - 1) ** 2
                if second_order == True:
                    loss += (y_output[0] - 1) ** 2
            # d1y_output = y_output.clone()
            # y_output[0] = 1
            d1y_output = (-y_output[0:-2] + y_output[2:]) / (2 * self.step)
            loss = loss + loss_mse(d1y_output + y_output[1:-1], self.x[1:-1])
            loss.backward()
            optimizer.step()
            # scheduler2.step()
            # scheduler.step()

            if i % 1000 == 0:
                print(i, loss.item())

        torch.save(model, 'Models/'+savefile)

    def plot(self,y_true,model_name='weak_loss_model.pt',filename_prefix='F'):

        if self.output_size == 1:
            y0 = [self.y0.item(), self.F(torch.tensor(0), self.y0).item()]  # Augment initial condition with intial slope
        else:
            y0 = self.y0.tolist() + self.F(torch.tensor(0), self.y0).tolist()
        model = torch.load('Models/'+model_name, map_location=torch.device('cpu'))

        y0 = torch.tensor(y0).to('cpu')

        if self.second_derivate_expanison==True:
            version = '_sde'
        else:
            version = '_sde_no'

        model.eval()
        # Plot solution
        plt.rcParams['font.size'] = 13
        with torch.no_grad():  # Tell torch to stop keeping track of gradients

            x = self.x.to('cpu')
            f = model(x)
            if self.second_derivate_expanison == True:
                net = self.y(model,x,y0).numpy()
            else:
                net = f.numpy()
            true = y_true(x).numpy()
            x = x.numpy()

            if self.numerical == False:
                compare_plot_legend = 'True Sol'
                compare_title = 'FinNet vs True'
                error_ylabel = 'x(t): |True - FinNet|'
                plotstart = 0
                corrector_plot_legend = 'True Corrector'
            else:
                compare_plot_legend = 'Num Sol'
                compare_title = 'FinNet vs Num'
                error_ylabel = 'x(t): |Num - FinNet|'
                plotstart = 30
                corrector_plot_legend = 'Num Corrector'

            plt.figure()
            for i in range(np.shape(net)[1]):
                plt.plot(x, net[:,i], label=self.plot_labels[i])
            for i in range(np.shape(net)[1]):
                plt.plot(x, true[:,i], color = 'k', label= compare_plot_legend if i == 0  else None, linestyle='--')
            plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
            plt.xlabel('t')
            plt.ylabel('x(t)')
            plt.title(compare_title)
            plt.savefig('Figures/'+filename_prefix+'_NeuralNetPlot_FinNet' + version + '.pdf', bbox_inches = "tight")
            plt.show()

            plt.figure()
            for i in range(np.shape(net)[1]):
                plt.plot(x, np.absolute(net[:,i] - true[:,i]),label='Error '+self.plot_labels[i])
            plt.xlabel('t')
            plt.ylabel(error_ylabel)
            plt.title('Error')
            plt.legend()
            plt.savefig('Figures/'+filename_prefix+'_NeuralNetErrorPlot_FinNet' + version + '.pdf', bbox_inches = "tight")
            plt.show()

            plt.figure()
            for i in range(np.shape(net)[1]):
                plt.plot(x[plotstart:], f.numpy()[plotstart:, i], label='Corrector ' + self.plot_labels[i])
            for i in range(np.shape(net)[1]):
                plt.plot(x[plotstart:],2*(true[plotstart:, i].reshape(-1, 1) - y0[i:i + 1].numpy() -
                        y0[self.output_size + i:self.output_size + i + 1].numpy()*x[plotstart:])/x[plotstart:]**2,
                    color='k', label=corrector_plot_legend if i == 0 else None, linestyle='--')
            # for i in range(np.shape(net)[1]):
            #     plt.plot(x, f.numpy()[:,i], label='Neural Net Corrector_'+str(i+1))
            # for i in range(np.shape(net)[1]):
            #     plt.plot(x, 2 * (true[:,i].reshape(-1,1) - y0[i:i+1].numpy() -
            #                      y0[self.output_size+i:self.output_size+i+1].numpy()*x) / x**2,
            #              label='True Corrector_'+str(i+1))
            plt.xlabel('t')
            plt.ylabel(r'$\xi$')
            plt.title('Neural Net Corrector')
            plt.legend()
            plt.savefig('Figures/'+filename_prefix+'_NeuralNetCorrector_FinNet' + version + '.pdf', bbox_inches = "tight")
            plt.show()
