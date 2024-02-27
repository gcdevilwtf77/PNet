"""Weak Loss for Solving ODEs with Neural Networks"""
import numpy as np
import torch
from pinns_utils_vectorized_class import ode
from pinns_numerical_sim import numerical_solutions

def F1(x,y):
    return -y + torch.cos(x) + torch.sin(x) 
def y1(x):
    return torch.sin(x) #True solution for F1
def F2(x,y):
    return  -y + x 
def y2(x):
    return x - 1 + 2*torch.exp(-x) #True solution for F2
def F3(x,y):
    return -y + torch.exp(-x) 
def y3(x):
    return (torch.cos(x) + torch.sin(x) + torch.exp(-x))/2 #True solution for F3
def F4(x,y):
    return -y**2 + torch.sin(x)**2 + torch.cos(x)
def y4(x):
    return torch.sin(x) #True solution for F4
def F5(x,y):
    return -y**2 + torch.cos(x)**2 - torch.sin(x)
def y5(x):
    return torch.cos(x) #True solution for F5
def F6(x,y):
    return 2*torch.sqrt(torch.relu(y))
def y6(x):
    return x**2 #True solution for F60
def F7(x,y):
    return 2*torch.sqrt(torch.relu(y)+1e-6)
def y7(x):
    return x**2 #True solution for F7
def F8(x,y):
    try:
       return torch.vstack([y[:,1],-y[:,0]]).T
    except:
        return torch.hstack([y[1], -y[0]])
def y8(x):
    return torch.hstack([torch.sin(x),torch.cos(x)]) #True solution for F8
def F9(x,y):
    try:
       return torch.vstack([y[:,1],-y[:,0] + torch.exp(-x)[:,0]]).T
    except:
        return torch.hstack([y[1], -y[0] + torch.exp(-x)])
def y9(x):
    return torch.hstack([(torch.cos(x) + torch.sin(x) + torch.exp(-x))/2,
                         (-torch.sin(x) + torch.cos(x) - torch.exp(-x))/2]) #True solution for F9
def F10(x,y):
    try:
       return torch.vstack([y[:,2],y[:,3],-y[:,0],-y[:,1]]).T
    except:
        return torch.hstack([y[2],y[3],-y[0],-y[1]])
def y10(x):
    return torch.hstack([torch.sin(x),torch.sin(x),torch.cos(x),torch.cos(x)]) #True solution for F10
def F11(x,y):
    try:
       return torch.vstack([y[:,2],y[:,3],-y[:,0]/(y[:,0]**2+y[:,1]**2)**(3/2),-y[:,1]/(y[:,0]**2+y[:,1]**2)**(3/2)]).T
    except:
        return torch.hstack([y[2],y[3],-y[0]/(y[0]**2+y[1]**2)**(3/2),-y[1]/(y[0]**2+y[1]**2)**(3/2)])
def F11_num(t,y):
    return [y[2],y[3],-y[0]/(y[0]**2+y[1]**2)**(3/2),-y[1]/(y[0]**2+y[1]**2)**(3/2)]
def y11(x):
    sim_output = numerical_solutions(F_num=F11_num,t0=-1,final_time_forward=1).numerical_integrate()
    if x.dim() == 0:
        return torch.from_numpy(sim_output)
    else:
        return torch.from_numpy(
                numerical_solutions(F_num=F11_num,t0=0,final_time_forward=x[-1],dt=x[-1]/len(x)).numerical_integrate(),
                                    y0=sim_output.tolist())
    # return torch.hstack([torch.sin(x),torch.sin(x),torch.cos(x),torch.cos(x)]) #True solution for F11
def F12(x,y):
    try:
       return torch.vstack([y[:,2],y[:,3],-y[:,0]/(y[:,0]**2+y[:,1]**2)**(3/2),-y[:,1]/(y[:,0]**2+y[:,1]**2)**(3/2)]).T
    except:
        return torch.hstack([y[2],y[3],-y[0]/(y[0]**2+y[1]**2)**(3/2),-y[1]/(y[0]**2+y[1]**2)**(3/2)])
def y12(x):
    return torch.hstack([torch.sin(x),torch.sin(x),torch.cos(x),torch.cos(x)]) #True solution for F12

def F20(x,y):
    try:
       return torch.vstack([y[:,1],y[:,0]*0]).T
    except:
        return torch.hstack([y[1],y[0]*0])
# [y[1],y[0]*1]
def y20(x):
    return torch.hstack([x*0,x*0]) #True solution for F10
# [x,1]
def F21(x,y):
    try:
       return torch.vstack([y[:,1],-y[:,0],]).T
    except:
        return torch.hstack([y[1],-y[0]])
def y21(x):
    return torch.hstack([torch.sin(x),torch.cos(x)]) #True solution for F10

def F23(x,y):
    try:
       return torch.vstack([y[:,1],y[:,0]*0]).T
    except:
        return torch.hstack([y[1],y[0]*0])
# [y[1],y[0]*1]
def y23(x):
    return torch.hstack([x*0 + 1,x*0]) #True solution for F10
# [x,1]

def F24(x,y):
    try:
       return torch.vstack([y[:,1],y[:,0]*0]).T
    except:
        return torch.hstack([y[1],y[0]*0])
# [y[1],y[0]*1]
def y24(x):
    return torch.hstack([x,x*0 + 1]) #True solution for F10
# [x,1]
F24_plot_labels = ['x','x_prime']

def F25(x,y):
    try:
       return torch.vstack([y[:,1],y[:,0]*0]).T
    except:
        return torch.hstack([y[1],y[0]*0])
# [y[1],y[0]*1]
def y25(x):
    return torch.hstack([x+1,x*0 + 1]) #True solution for F10
# [x,1]
F25_plot_labels = ['x','x_prime']

def F26(x,y):
    try:
       return torch.vstack([y[:,1],y[:,0]*0 + 1]).T
    except:
        return torch.hstack([y[1],y[0]*0 + 1])
def y26(x):
    return torch.hstack([1/2*x**2,x]) #True solution for F26
F26_plot_labels = ['x','x_prime']

zero = torch.tensor(0)
number_reference = '26'
name = 'F' + number_reference
model_name = name + '_model_PINNS.pt'
F = locals()['F'+number_reference]
y = locals()['y'+number_reference]

try:
    locals()['F' + number_reference + '_num']
    numerical=True
except:
    numerical=False

try:
    plot_labels = locals()['F'+ number_reference + '_plot_labels']
except:
    plot_labels = ['x']

output = ode(F,y(zero),zero,1,epochs=int(1e5),batch_size=1000 ,lr=0.1,num_hidden=100,
             numerical=numerical,second_derivate_expanison=True,plot_labels=plot_labels)
output.train(model_name)
output.plot(y,model_name,name)