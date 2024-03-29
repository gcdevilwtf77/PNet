import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os

#Our neural network as 2 layers with 100 hidden nodes 
class Net(nn.Module):
    def __init__(self, n_hidden=100,output_size=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,output_size)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ode(object):
    def __init__(self,F,y0,T,rule='trapezoid',epochs=10000,lr=0.01,batch_size=1000,cuda=True,num_hidden=100,
                 numerical=False,record_detailed=False,plots_detailed=False,plot_labels=['x'],y_limit=0,y_true='None'):
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
        self.record_detailed = record_detailed
        self.plots_detailed = plots_detailed
        self.y_limit = y_limit
        self.y_true = y_true

        if plot_labels[0] == 'x' and len(plot_labels) == 1:
            self.plot_labels = []
            for i in range(np.size(y0.numpy())):
                self.plot_labels.append(plot_labels[0] + '_' + str(i+1))
        else:
            self.plot_labels=plot_labels

        # Set up device and model
        self.use_cuda = cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = Net(self.num_hidden, self.output_size).to(self.device)

        # Set up x
        self.dx = self.T/self.batch_size
        self.x = torch.arange(self.dx/2, self.T, self.dx).reshape((self.batch_size, 1)).to(self.device)

        torch.set_default_dtype(torch.float64)

    def y(self, model,x,y0):
        if self.output_size == 1:
            return y0[0] + y0[1] * x + (1 / 2) * model(x) * x ** 2
        else:
           return y0[:self.output_size] + y0[self.output_size:]*x + (1/2)*model(x)*x**2

    def integrate(self,model,y0):

        x_left = self.x - self.dx/2
        x_right = self.x + self.dx/2
        y_left, y_mid, y_right = self.y(model,x_left,y0), self.y(model,self.x,y0), self.y(model,x_right,y0)
        F_left, F_mid, F_right = self.F(x_left,y_left), self.F(self.x,y_mid), self.F(x_right,y_right)
        if self.rule == 'midpoint':
            return self.dx*torch.cumsum(F_mid, dim=0)
        if self.rule == 'leftpoint':
            return self.dx*torch.cumsum(F_left, dim=0)
        if self.rule == 'rightpoint':
            return self.dx*torch.cumsum(F_right, dim=0)
        if self.rule == 'simpson':
            return self.dx*torch.cumsum((F_left + 4*F_mid + F_right)/6, dim=0)
        if self.rule == 'trapezoid':
            return self.dx*torch.cumsum((F_left + F_right)/2, dim=0)

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

        model.train()
        if self.record_detailed == True:
            import pandas as pd
            x = self.x.to('cpu')
            true = self.y_true(x).numpy()
            recording_iteration_mod = np.min(np.array([self.epochs/1e2,1e6]))

        start = time()
        for i in range(self.epochs):

            optimizer.zero_grad()
            loss = self.dx*torch.sum(torch.abs(self.y(model,self.x+self.dx/2,y0) - y0[:self.output_size] \
                                               - self.integrate(model,y0)))
            # loss = torch.tanh(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if self.record_detailed == True:
                if os.path.isdir(self.F.__name__) == False:
                    os.mkdir(self.F.__name__)
                    os.mkdir(self.F.__name__+'\\Data')
                    os.mkdir(self.F.__name__+'\\Figures')
                    os.mkdir(self.F.__name__+'\\Models')
                if i%recording_iteration_mod == 0 or (i+1)==self.epochs:
                    if i==0:
                        header='True'
                        mode='w'
                    else:
                        header='False'
                        mode='a'
                    # data creation and save
                    detailed_data = pd.DataFrame()
                    detailed_data['iteration'] = i
                    detailed_data['loss'] = loss.item()
                    detailed_data['time_taken'] = (time()-start)/60**2
                    detailed_data = detailed_data[['iteration','loss','time_taken']]
                    detailed_data.to_csv(self.F.__name__+'/Data/detailed_data_record_'+ self.F.__name__ +
                                         '_PiNNs.csv',index=False,header=header,mode=mode)

                    # model save
                    torch.save(model, self.F.__name__+'/Models/'+ self.F.__name__ +
                                         '_model_PiNNs_iteration_' + str(i) + '.pt')

                    f = model(self.x)
                    net = self.y(model, self.x, y0).to('cpu').detach().numpy()

                    if self.numerical == False:
                        compare_plot_legend = 'True Sol'
                        compare_title = 'PiNNs vs True'
                        error_ylabel = 'x(t): |True - PiNNs|'
                        corrector_plot_legend = 'True Corrector'
                    else:
                        compare_plot_legend = 'Num Sol'
                        compare_title = 'PiNNs vs Num'
                        error_ylabel = 'x(t): |Num - PiNNs|'
                        corrector_plot_legend = 'Num Corrector'

                    for j in range(np.shape(net)[1]):
                        plt.figure()
                        plt.plot(x, net[:, j], label=self.plot_labels[j])
                        plt.plot(x, true[:, j], color='k', label=compare_plot_legend,linestyle='--')
                        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        plt.xlabel('t')
                        plt.ylabel('x(t)')
                        plt.title(compare_title)
                        plt.savefig(self.F.__name__+'/Figures/'+ self.F.__name__ +
                                         '_NeuralNetPlot_PINNS_'+ str(self.plot_labels[j]) + '_iteration_'
                                    + str(i) + '.pdf', bbox_inches="tight")
                        plt.close()

                        plt.figure()
                        plt.plot(x, np.absolute(net[:, j] - true[:, j]), label='Error ' + self.plot_labels[j])
                        plt.xlabel('t')
                        plt.ylabel(error_ylabel)
                        plt.title('Error')
                        plt.legend()
                        plt.savefig(self.F.__name__+'/Figures/'+ self.F.__name__ +
                                         '_NeuralNetErrorPlot_PINNS_'+ str(self.plot_labels[j]) + '_iteration_'
                                    + str(i) + '.pdf',
                                    bbox_inches="tight")
                        plt.close()

                        plt.figure()
                        plt.plot(x, f.to('cpu').detach().numpy()[:, j], label='Corrector ' + self.plot_labels[j])
                        plt.plot(x, 2 * (true[:, j].reshape(-1, 1) - y0[j:j + 1].to('cpu').detach().numpy() -
                                    y0[self.output_size + j:self.output_size + j + 1].to('cpu').detach().numpy()*\
                                         x.to('cpu').detach().numpy())/x.to('cpu').detach().numpy()**2,
                                 color='k', label=corrector_plot_legend, linestyle='--')
                        plt.xlabel('t')
                        plt.ylabel(r'$\xi$')
                        plt.title('Neural Net Corrector')
                        plt.legend()
                        plt.savefig(self.F.__name__+'/Figures/'+ self.F.__name__ +
                                         '_NeuralNetCorrector_PINNS_'+ str(self.plot_labels[j])
                                    + '_iteration_' + str(i) + '.pdf',
                                    bbox_inches="tight")
                        plt.close()

            if i % (self.epochs/10) == 0: #1000 == 0:
                print(i, loss.item())
        # print('NN time: ' + str(time()-start))
        torch.save(model, 'Models/'+savefile)

    def plot(self,y_true,model_name='weak_loss_model.pt',filename_prefix='F'):

        if self.output_size == 1:
            y0 = [self.y0.item(), self.F(torch.tensor(0), self.y0).item()]  # Augment initial condition with intial slope
        else:
            y0 = self.y0.tolist() + self.F(torch.tensor(0), self.y0).tolist()
        model = torch.load('Models/'+model_name, map_location=torch.device('cpu'))

        y0 = torch.tensor(y0).to('cpu')

        model.eval()
        # Plot solution
        plt.rcParams['font.size'] = 13
        with torch.no_grad():  # Tell torch to stop keeping track of gradients

            x = self.x.to('cpu')
            f = model(x)
            net = self.y(model,x,y0).numpy()
            true = y_true(x).numpy()
            x = x.numpy()

            if self.numerical == False:
                compare_plot_legend = 'True Sol'
                compare_title = 'PNet vs True'
                error_ylabel = 'x(t): |True - PNet|'
                plotstart = 0
                corrector_plot_legend = 'True Corrector'
            else:
                compare_plot_legend = 'Num Sol'
                compare_title = 'PNet vs Num'
                error_ylabel = 'x(t): |Num - PNet|'
                plotstart = 0
                corrector_plot_legend = 'Num Corrector'

            if self.plots_detailed==False:

                plt.figure()
                for i in range(np.shape(net)[1]):
                    plt.plot(x, net[:,i], label=self.plot_labels[i])
                for i in range(np.shape(net)[1]):
                    plt.plot(x, true[:,i], color = 'k', label= compare_plot_legend if i == 0  else None, linestyle='--',
                             alpha=0.5,dashes=(5,5))
                plt.xlabel('t')
                plt.ylabel('x(t)')
                plt.title(compare_title)
                plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
                plt.savefig('Figures/'+filename_prefix+'_NeuralNetPlot.pdf', bbox_inches = "tight")
                plt.show()

                plt.figure()
                for i in range(np.shape(net)[1]):
                    plt.plot(x, np.absolute(net[:,i] - true[:,i]),label='Error '+self.plot_labels[i])
                plt.xlabel('t')
                plt.ylabel(error_ylabel)
                plt.title('Error')
                plt.legend()
                plt.savefig('Figures/'+filename_prefix+'_NeuralNetErrorPlot.pdf', bbox_inches = "tight")
                plt.show()

                plt.figure()
                # plotstart = 30
                for i in range(np.shape(net)[1]):
                    plt.plot(x[plotstart:], f.numpy()[plotstart:,i], label='Corrector '+self.plot_labels[i])
                for i in range(np.shape(net)[1]):
                    plt.plot(x[plotstart:], 2 * (true[plotstart:,i].reshape(-1,1) - y0[i:i+1].numpy() -
                                     y0[self.output_size+i:self.output_size+i+1].numpy()*x[plotstart:]) / x[plotstart:]**2,
                             color = 'k', label=corrector_plot_legend if i==0 else None, linestyle='--',
                             alpha=0.5,dashes=(5,5))
                if self.y_limit!=0:
                    if self.y_limit == 'Null':
                        y_limits = np.min([np.max(np.abs(true)),np.max(np.abs(2 * (true[plotstart:,i].reshape(-1,1) - y0[i:i+1].numpy() -
                                     y0[self.output_size+i:self.output_size+i+1].numpy()*x[plotstart:]) / x[plotstart:]**2))])
                    else:
                        y_limits = self.y_limit
                    plt.ylim(-y_limits,y_limits)
                plt.xlabel('t')
                plt.ylabel(r'$\xi$')
                plt.title('Neural Net Corrector')
                plt.legend()
                plt.savefig('Figures/'+filename_prefix+'_NeuralNetCorrector.pdf', bbox_inches = "tight")
                plt.show()

            elif self.plots_detailed==True:

                for i in range(np.shape(net)[1]):
                    plt.figure()

                    plt.plot(x, net[:,i], label=self.plot_labels[i])
                    plt.plot(x, true[:,i], color = 'k', label= compare_plot_legend, linestyle='--',
                             alpha=0.5,dashes=(5,5))
                    plt.xlabel('t')
                    plt.ylabel('x(t)')
                    plt.title(compare_title)
                    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
                    plt.savefig('Figures/'+filename_prefix+'_NeuralNetPlot_'+str(self.plot_labels[i])+'.pdf',
                                bbox_inches = "tight")
                    # plt.show()

                    plt.figure()
                    plt.plot(x, np.absolute(net[:,i] - true[:,i]),label='Error '+self.plot_labels[i])
                    plt.xlabel('t')
                    plt.ylabel(error_ylabel)
                    plt.title('Error')
                    plt.legend()
                    plt.savefig('Figures/'+filename_prefix+'_NeuralNetErrorPlot_'+str(self.plot_labels[i])+'.pdf',
                                bbox_inches = "tight")
                    # plt.show()

                    plt.figure()
                    # plotstart = 30
                    plt.plot(x[plotstart:], f.numpy()[plotstart:,i], label='Corrector '+self.plot_labels[i])
                    plt.plot(x[plotstart:], 2 * (true[plotstart:,i].reshape(-1,1) - y0[i:i+1].numpy() -
                                         y0[self.output_size+i:self.output_size+i+1].numpy()*x[plotstart:]) / x[plotstart:]**2,
                                 color = 'k', label=corrector_plot_legend, linestyle='--',
                                 alpha=0.5,dashes=(5,5))
                    plt.xlabel('t')
                    plt.ylabel(r'$\xi$')
                    plt.title('Neural Net Corrector')
                    plt.legend()
                    plt.savefig('Figures/'+filename_prefix+'_NeuralNetCorrector_'+str(self.plot_labels[i])+'.pdf',
                                bbox_inches = "tight")
                    # plt.show()
