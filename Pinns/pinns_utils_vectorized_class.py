import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import autograd
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os

#Our neural network as 2 layers with 100 hidden nodes 
class Net(nn.Module):
    def __init__(self, n_hidden=100,output_size=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,n_hidden)
        self.fc4 = nn.Linear(n_hidden,output_size)
        # self.fc2 = nn.Linear(n_hidden,n_hidden)
        # self.fc3 = nn.Linear(n_hidden, n_hidden)
        # self.fc4 = nn.Linear(n_hidden,n_hidden)
        # self.fc5 = nn.Linear(n_hidden,output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc4(x)
        # x = func.tanh((self.fc1(x)))
        # x = func.tanh(self.fc2(x))
        # x = func.tanh(self.fc3(x))
        # x = func.tanh(self.fc4(x))
        # x = self.fc5(x)
        return x

class ode(object):
    def __init__(self,F,y0,t0,T,rule='trapezoid',epochs=10000,lr=0.01,batch_size=1000,cuda=True,num_hidden=100,
                 numerical=False,second_derivate_expanison=True,record_detailed=False,plots_detailed=False,
                 plot_labels=['x'],y_true='None'):

        torch.set_default_dtype(torch.float64)

        self.F = F
        self.y0 = y0
        self.t0= t0
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
        self.record_detailed = record_detailed
        self.plots_detailed = plots_detailed
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
        self.step = (T - 0)/self.batch_size
        self.x = torch.arange(self.dx/2, self.T, self.dx).reshape((self.batch_size, 1)).to(self.device)
        # self.x = torch.arange(0, self.T + self.step, step=self.step).reshape((self.batch_size + 1, 1)).to(self.device)
        # self.x = torch.arange(0,self.T,self.step).unsqueeze(1).to(self.device)

    def y(self, model,x,y0):
        if self.output_size == 1:
            return y0[0] + y0[1] * x + (1 / 2) * model(x) * x ** 2
        else:
           return y0[:self.output_size] + y0[self.output_size:]*x + (1/2)*model(x)*x**2

    def train(self,savefile='model.pt'):


        if self.output_size == 1:
            y0 = [self.y0.item(), self.F(self.t0, self.y0).item()]  #Augment initial condition with intial slope
        else:
            y0 = self.y0.tolist() + self.F(self.t0, self.y0).tolist()

        model = self.model

        # if output_size > 1:
        y0 = torch.tensor(y0).to(self.device)

        #Set up optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=self.lr)#, weight_decay=0.0001)  #Learning rate
        scheduler = StepLR(optimizer, step_size=1, gamma=0.001**(1/(self.epochs)))
        # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        model.train()
        if self.record_detailed == True:
            import pandas as pd
            x = self.x.to('cpu')
            true = self.y_true(x).numpy()
            recording_iteration_mod = np.min(np.array([self.epochs/1e2,1e6]))#to determine how often to save things

        start = time()
        for i in range(self.epochs):

            optimizer.zero_grad()
            # x_grad = self.x+self.dx/2
            x_grad = self.x#[1:-1]
            # x_grad = self.T*torch.rand(self.x.size()[0]).unsqueeze(1).to(self.device)
            x_grad.requires_grad_()
            if self.second_derivate_expanison == True:
                y_output = self.y(model,x_grad,y0)
            else:
                y_output = model(x_grad)

            if self.F.__name__ == 'F2':
                dy_dx = torch.autograd.grad(y_output.sum(),x_grad,create_graph=True)[0].flatten()
                if self.second_derivate_expanison == True:
                    loss = torch.mean((dy_dx+y_output.flatten()-x_grad.flatten())**2)
                else:
                    loss = torch.mean((dy_dx+y_output.flatten()-x_grad.flatten())**2 +\
                           (y_output[0]-y0[0].flatten())**2)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if self.F.__name__ == 'F3':
                dy_dx = torch.autograd.grad(y_output.sum(), x_grad, create_graph=True)[0].flatten()
                dy_dxx = torch.autograd.grad(dy_dx.sum(), x_grad, create_graph=True)[0].flatten()
                if self.second_derivate_expanison == True:
                    loss = torch.mean((dy_dxx+y_output.flatten()-torch.exp(-x_grad).flatten())**2)
                else:
                    loss = torch.mean((dy_dxx+y_output.flatten()-torch.exp(-x_grad).flatten())**2 + \
                                      (y_output[0]-y0[0].flatten())**2)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if self.F.__name__ == 'F9':
                u = y_output[:,0]
                uprime = y_output[:,1]
                u_x = torch.autograd.grad(u.sum(), x_grad, create_graph=True)[0].flatten()
                uprime_x = torch.autograd.grad(uprime.sum(), x_grad, create_graph=True)[0].flatten()
                loss_u = (uprime_x + u - torch.exp(-x_grad).flatten()) ** 2
                loss_uprime = (uprime - u_x) ** 2
                if self.second_derivate_expanison == True:
                    loss = torch.mean(loss_u + loss_uprime)
                else:
                    loss_boundary_u = (u[0]-y0[0].flatten())**2 + (u_x[0]-y0[2].flatten())**2
                    loss_boundary_uprime = (uprime[0]-y0[1].flatten())**2 + (uprime_x[0]-y0[3].flatten())**2
                    loss = torch.mean( loss_u + loss_uprime + loss_boundary_u + loss_boundary_uprime)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if self.F.__name__== 'F10':
                q1 = y_output[:,0]
                q2 = y_output[:,1]
                p1 = y_output[:,2]
                p2 = y_output[:,3]
                q1_t = torch.autograd.grad(q1.sum(), x_grad, create_graph=True)[0].flatten()
                q2_t = torch.autograd.grad(q2.sum(), x_grad, create_graph=True)[0].flatten()
                p1_t = torch.autograd.grad(p1.sum(), x_grad, create_graph=True)[0].flatten()
                p2_t = torch.autograd.grad(p2.sum(), x_grad, create_graph=True)[0].flatten()
                loss_q1 = (q1_t - p1) **2
                loss_q2 = (q2_t - p2) ** 2
                loss_p1 = (p1_t + q1) **2
                loss_p2 = (p2_t + q2) ** 2
                if self.second_derivate_expanison == True:
                    loss = torch.mean(loss_q1 + loss_q2 + loss_p1 + loss_p2)
                else:
                    loss_ic = (q1[0] - y0[0])**2 + (q2[0] - y0[1])**2 + (p1[0] - y0[2])**2 + (p2[0] - y0[3])**2
                    loss = loss_q1 + loss_q2 + loss_p1 + loss_p2 + loss_ic
                loss.backward()
                optimizer.step()
                scheduler.step()
            # # F10 second derivative false
            # q1 = y_output[:,0]
            # q2 = y_output[:,1]
            # p1 = y_output[:,2]
            # p2 = y_output[:,3]
            # q1_t = torch.autograd.grad(q1.sum(), x_grad, create_graph=True)[0].flatten()
            # q2_t = torch.autograd.grad(q2.sum(), x_grad, create_graph=True)[0].flatten()
            # p1_t = torch.autograd.grad(p1.sum(), x_grad, create_graph=True)[0].flatten()
            # p2_t = torch.autograd.grad(p2.sum(), x_grad, create_graph=True)[0].flatten()
            # loss_q1 = torch.mean((q1_t - p1) **2)
            # loss_q2 = torch.mean((q2_t - p2) **2)
            # loss_p1 = torch.mean((p1_t + q1) **2)
            # loss_p2 = torch.mean((p2_t + q2) **2)
            # loss_ic = (q1[0] - y0[0])**2 + (q2[0] - y0[1])**2 + (p1[0] - y0[2])**2 + (p2[0] - y0[3])**2
            # loss = loss_q1 + loss_q2 + loss_p1 + loss_p2 + loss_ic
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            if self.F.__name__ == 'F19':
                k1 = 0.04
                k2 = 3*1e7
                k3 = 1e4
                y1 = y_output[:, 0]
                y2 = y_output[:, 1]
                y3 = y_output[:, 2]
                y1_t = torch.autograd.grad(y1.sum(), x_grad, create_graph=True)[0].flatten()
                y2_t = torch.autograd.grad(y2.sum(), x_grad, create_graph=True)[0].flatten()
                y3_t = torch.autograd.grad(y3.sum(), x_grad, create_graph=True)[0].flatten()
                loss_y1 = (y1_t + k1*y1 - k3*y2*y3)**2
                loss_y2 = (y2_t - k1*y1 + k2*y2**2 + k3*y2*y3)**2
                loss_y3 = (y3_t - k2*y2**2)**2
                loss = torch.mean(loss_y1 + loss_y2 + loss_y3)
                loss.backward()
                optimizer.step()
            if self.F.__name__ == 'F20':
                u = y_output[:, 0]
                uprime = y_output[:, 1]
                u_x = torch.autograd.grad(u.sum(), x_grad, create_graph=True)[0]
                uprime_x = torch.autograd.grad(uprime.sum(), x_grad, create_graph=True)[0]
                loss = torch.mean((uprime - u_x) ** 2 + (uprime_x - 0*u)**2)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if self.F.__name__ == 'F21':
                u = y_output[:, 0]
                uprime = y_output[:, 1]
                u_x = torch.autograd.grad(u.sum(), x_grad, create_graph=True)[0].flatten()
                uprime_x = torch.autograd.grad(uprime.sum(), x_grad, create_graph=True)[0].flatten()
                loss = torch.mean((u_x - uprime) ** 2 + (uprime_x + u) ** 2)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if self.F.__name__ == 'F22':
                q = y_output[:,0]
                p = y_output[:,1]
                q_t = torch.autograd.grad(q.sum(), x_grad, create_graph=True)[0].flatten()
                p_t = torch.autograd.grad(p.sum(), x_grad, create_graph=True)[0].flatten()
                loss_q = (q_t - p) **2
                loss_p = (p_t + 0.05*p + 9.81*torch.sin(q).flatten()) **2
                if self.second_derivate_expanison == True:
                    loss = torch.mean(loss_q + loss_p)
                else:
                    loss_ic = (q[0].flatten() - y0[0].flatten())**2 + (p[0].flatten() - y0[1].flatten())**2
                    loss = torch.mean(loss_q + loss_p + loss_ic)
                loss.backward()
                optimizer.step()
                scheduler.step()
            # # F22 second derivative false
            # q = y_output[:, 0]
            # p = y_output[:, 1]
            # q_t = torch.autograd.grad(q.sum(), x_grad, create_graph=True)[0].flatten()
            # p_t = torch.autograd.grad(p.sum(), x_grad, create_graph=True)[0].flatten()
            # loss_q = (q_t - p) ** 2
            # loss_p = (p_t + 0.05 * p + 9.81 * torch.sin(q).flatten()) ** 2
            # loss_ic = (q[0].flatten() - y0[0].flatten())**2 + (p[0].flatten() - y0[1].flatten())**2
            # loss = torch.mean(loss_q + loss_p + loss_ic)
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            if self.F.__name__ == 'F23':
                u = y_output[:, 0]
                uprime = y_output[:, 1]
                u_x = torch.autograd.grad(u.sum(), x_grad, create_graph=True)[0].flatten()
                uprime_x = torch.autograd.grad(uprime.sum(), x_grad, create_graph=True)[0].flatten()
                loss = torch.mean((uprime - u_x) ** 2 + (uprime_x - 0 * u) ** 2)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if self.F.__name__ == 'F24':
                u = y_output[:, 0]
                uprime = y_output[:, 1]
                u_x = torch.autograd.grad(u.sum(), x_grad, create_graph=True)[0].flatten()
                uprime_x = torch.autograd.grad(uprime.sum(), x_grad, create_graph=True)[0].flatten()
                loss = torch.mean((uprime - u_x) ** 2 + (uprime_x - 0*u) ** 2)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if self.F.__name__ == 'F25':
                u = y_output[:, 0]
                uprime = y_output[:, 1]
                u_x = torch.autograd.grad(u.sum(), x_grad, create_graph=True)[0].flatten()
                uprime_x = torch.autograd.grad(uprime.sum(), x_grad, create_graph=True)[0].flatten()
                loss = torch.mean((uprime - u_x) ** 2 + (uprime_x - 0*u) ** 2)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if self.F.__name__ == 'F26':
                u = y_output[:, 0]

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
                    # detailed_data = pd.DataFrame(np.array([y1.to('cpu').detach().numpy(),y2.to('cpu').detach().numpy(),
                    #                              y3.to('cpu').detach().numpy(),y1_t.to('cpu').detach().numpy(),
                    #                              y2_t.to('cpu').detach().numpy(),y3_t.to('cpu').detach().numpy(),
                    #                              (y1_t + k1*y1 - k3*y2*y3).to('cpu').detach().numpy(),
                    #                              (y2_t - k1*y1 + k2*y2 + k3*y2*y3).to('cpu').detach().numpy(),
                    #                              (y3_t - k2*y2).to('cpu').detach().numpy()]).T,columns=['y1','y2','y3',
                    #                                                                                      'y1_t','y2_t',
                    #                                                                                      'y3_t','y1_ode',
                    #                                                                                      'y2_ode','y3_ode'])
                    # detailed_data['iteration'] = i
                    # detailed_data['loss'] = loss.item()
                    # detailed_data['time_taken'] = (time()-start)/60**2
                    # detailed_data = detailed_data[['iteration','loss','time_taken','y1','y2','y3','y1_t','y2_t','y3_t',
                    #                                'y1_ode','y2_ode','y3_ode']]
                    detailed_data = pd.DataFrame()
                    detailed_data['iteration'] = i
                    detailed_data['loss'] = loss.item()
                    detailed_data['time_taken'] = (time() - start) / 60 ** 2
                    detailed_data = detailed_data[
                        ['iteration', 'loss', 'time_taken']]
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





            # scheduler1.step()
            # scheduler2.step()

            if i % int(self.epochs/10) == 0:
                print(i, loss.item())
                # print(i, 'loss_u:' + str(loss_u.mean().item()), 'loss_uprime:' + str(loss_uprime.mean().item()))

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
                net = self.y(model, x, y0).numpy()
            else:
                net = f.numpy()
            true = y_true(x).numpy()
            x = x.numpy()

            if self.numerical == False:
                compare_plot_legend = 'True Sol'
                compare_title = 'PiNNs vs True'
                error_ylabel = 'x(t): |True - PiNNs|'
                plotstart = 0
                corrector_plot_legend = 'True Corrector'
            else:
                compare_plot_legend = 'Num Sol'
                compare_title = 'PiNNs vs Num'
                error_ylabel = 'x(t): |Num - PiNNs|'
                plotstart = 30
                corrector_plot_legend = 'Num Corrector'

            if self.plots_detailed == False:

                plt.figure()
                for i in range(np.shape(net)[1]):
                    plt.plot(x, net[:, i], label=self.plot_labels[i])
                for i in range(np.shape(net)[1]):
                    plt.plot(x, true[:, i], color='k', label=compare_plot_legend if i == 0 else None, linestyle='--',
                             alpha=0.5, dashes=(5, 5))
                plt.xlabel('t')
                plt.ylabel('x(t)')
                plt.title(compare_title)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig('Figures/' + filename_prefix + '_NeuralNetPlot.pdf', bbox_inches="tight")
                plt.show()

                plt.figure()
                for i in range(np.shape(net)[1]):
                    plt.plot(x, np.absolute(net[:, i] - true[:, i]), label='Error ' + self.plot_labels[i])
                plt.xlabel('t')
                plt.ylabel(error_ylabel)
                plt.title('Error')
                plt.legend()
                plt.savefig('Figures/' + filename_prefix + '_NeuralNetErrorPlot.pdf', bbox_inches="tight")
                plt.show()

                plt.figure()
                # plotstart = 30
                for i in range(np.shape(net)[1]):
                    plt.plot(x[plotstart:], f.numpy()[plotstart:, i], label='Corrector ' + self.plot_labels[i])
                for i in range(np.shape(net)[1]):
                    plt.plot(x[plotstart:], 2 * (true[plotstart:, i].reshape(-1, 1) - y0[i:i + 1].numpy() -
                                                 y0[self.output_size + i:self.output_size + i + 1].numpy() * x[
                                                                                                             plotstart:]) / x[
                                                                                                                            plotstart:] ** 2,
                             color='k', label=corrector_plot_legend if i == 0 else None, linestyle='--',
                             alpha=0.5, dashes=(5, 5))
                plt.xlabel('t')
                plt.ylabel(r'$\xi$')
                plt.title('Neural Net Corrector')
                plt.legend()
                plt.savefig('Figures/' + filename_prefix + '_NeuralNetCorrector.pdf', bbox_inches="tight")
                plt.show()

            elif self.plots_detailed == True:

                for i in range(np.shape(net)[1]):
                    plt.figure()

                    plt.plot(x, net[:, i], label=self.plot_labels[i])
                    plt.plot(x, true[:, i], color='k', label=compare_plot_legend, linestyle='--',
                             alpha=0.5, dashes=(5, 5))
                    plt.xlabel('t')
                    plt.ylabel('x(t)')
                    plt.title(compare_title)
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.savefig('Figures/' + filename_prefix + '_NeuralNetPlot_' + str(self.plot_labels[i]) + '.pdf',
                                bbox_inches="tight")
                    # plt.show()

                    plt.figure()
                    plt.plot(x, np.absolute(net[:, i] - true[:, i]), label='Error ' + self.plot_labels[i])
                    plt.xlabel('t')
                    plt.ylabel(error_ylabel)
                    plt.title('Error')
                    plt.legend()
                    plt.savefig(
                        'Figures/' + filename_prefix + '_NeuralNetErrorPlot_' + str(self.plot_labels[i]) + '.pdf',
                        bbox_inches="tight")
                    # plt.show()

                    plt.figure()
                    # plotstart = 30
                    plt.plot(x[plotstart:], f.numpy()[plotstart:, i], label='Corrector ' + self.plot_labels[i])
                    plt.plot(x[plotstart:], 2 * (true[plotstart:, i].reshape(-1, 1) - y0[i:i + 1].numpy() -
                                                 y0[self.output_size + i:self.output_size + i + 1].numpy() * x[
                                                                                                             plotstart:]) / x[
                                                                                                                            plotstart:] ** 2,
                             color='k', label=corrector_plot_legend, linestyle='--',
                             alpha=0.5, dashes=(5, 5))
                    plt.xlabel('t')
                    plt.ylabel(r'$\xi$')
                    plt.title('Neural Net Corrector')
                    plt.legend()
                    plt.savefig(
                        'Figures/' + filename_prefix + '_NeuralNetCorrector_' + str(self.plot_labels[i]) + '.pdf',
                        bbox_inches="tight")
                    # plt.show()
