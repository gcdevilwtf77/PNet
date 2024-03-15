"""Weak Loss for Solving ODEs with Neural Networks"""
import numpy as np
from scipy.integrate import odeint
import torch
from utils_vectorized_class import ode
from numerical_sim import numerical_solutions

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
    return x**2 #True solution for F6
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
F8_plot_labels = ['q_1','p_1']

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
# def F10(x,y):
#     try:
#        return torch.vstack([y[:,2],y[:,3],-y[:,0],-y[:,1]]).T
#     except:
#         return torch.hstack([y[2],y[3],-y[0],-y[1]])
# def y10(x):
#     return torch.hstack([torch.sin(x),torch.sin(x),torch.cos(x),torch.cos(x)]) #True solution for F10
F10_plot_labels = ['q_1','q_2','p_1','p_2']

def F11(x,y):
    try:
       return torch.vstack([y[:,2],y[:,3],-y[:,0]/(y[:,0]**2+y[:,1]**2)**(3/2),-y[:,1]/(y[:,0]**2+y[:,1]**2)**(3/2)]).T
    except:
        return torch.hstack([y[2],y[3],-y[0]/(y[0]**2+y[1]**2)**(3/2),-y[1]/(y[0]**2+y[1]**2)**(3/2)])
def F11_num(t,y):
    return [y[2],y[3],-y[0]/(y[0]**2+y[1]**2)**(3/2),-y[1]/(y[0]**2+y[1]**2)**(3/2)]
def y11(x):
    sim_output = numerical_solutions(F_num=F11_num, t0=-1, final_time_forward=1,y0=(-1,-2,-10,-11)).numerical_integrate()
    if x.dim() == 0:
        return torch.from_numpy(sim_output)
    else:
        return torch.from_numpy(
            numerical_solutions(F_num=F11_num, t0=0, final_time_forward=x[-1], dt=x[-1] / len(x),
                                y0=sim_output.tolist()).numerical_integrate())#True solution for F11
F11_plot_labels = ['q_1','q_2','p_1','p_2']
def F12(x,y):
    try:
       return torch.vstack([y[:,3],y[:,4],y[:,5],-y[:,0]/(y[:,0]**2+y[:,1]**2+y[:,2]**2)**(3/2),
                            -y[:,1]/(y[:,0]**2+y[:,1]**2+y[:,2]**2)**(3/2),
                            -y[:,2]/(y[:,0]**2+y[:,1]**2+y[:,2]**2)**(3/2)]).T
    except:
        return torch.hstack([y[3],y[4],y[5],-y[0]/(y[0]**2+y[1]**2+y[2]**2)**(3/2),-y[1]/(y[0]**2+y[1]**2+y[2]**2)**(3/2),
            -y[2]/(y[0]**2+y[1]**2+y[2])**(3/2)])
def F12_num(t,y):
    return [y[3],y[4],y[5],-y[0]/(y[0]**2+y[1]**2+y[2]**2)**(3/2),-y[1]/(y[0]**2+y[1]**2+y[2]**2)**(3/2),
            -y[2]/(y[0]**2+y[1]**2+y[2]**2)**(3/2)]

# H = 2*(p_1^2 + p_2^2 + p_3^2) - 1/|| q||
def y12(x):
    sim_output = numerical_solutions(F_num=F12_num, t0=-1, final_time_forward=1,
                                     y0=(-1,-2,-3,-0.1,-.11,-.12)).numerical_integrate()
    if x.dim() == 0:
        return torch.from_numpy(sim_output)
    else:
        return torch.from_numpy(
            numerical_solutions(F_num=F12_num, t0=0, final_time_forward=x[-1], dt=x[-1] / (len(x)),
                                y0=sim_output.tolist()).numerical_integrate()) #True solution for F12
F12_plot_labels = ['q_1','q_2','q_3','p_1','p_2','p_3']

def F13(x,y):
    try:
       return torch.vstack([y[:,2],y[:,3],y[:,0]/((y[:,1]-y[:,0])**2)**(3/2),-y[:,1]/((y[:,1]-y[:,0])**2)**(3/2)]).T
    except:
        return torch.hstack([y[2],y[3],y[0]/((y[1]-y[0])**2)**(3/2),-y[1]/((y[1]-y[0])**2)**(3/2)])
def F13_num(t,y):
    return [y[2],y[3],y[0]/((y[1]-y[0])**2)**(3/2),-y[1]/((y[1]-y[0])**2)**(3/2)]
def y13(x):
    sim_output = numerical_solutions(F_num=F13_num, t0=-1, final_time_forward=1,y0=(1,2,.1,11)).numerical_integrate()
    if x.dim() == 0:
        return torch.from_numpy(sim_output)
    else:
        return torch.from_numpy(
            numerical_solutions(F_num=F13_num, t0=0, final_time_forward=x[-1], dt=x[-1] / len(x),
                                y0=sim_output.tolist()).numerical_integrate())#True solution for F11
F13_plot_labels = ['q_1','q_2','p_1','p_2']

def F14(x,y):
    try:
       return torch.vstack([y[:,2],y[:,3],1/((y[:,1]-y[:,0])**2),-1/((y[:,1]-y[:,0])**2)]).T
    except:
        return torch.hstack([y[2],y[3],1/((y[1]-y[0])**2),-1/((y[1]-y[0])**2)])
def F14_num(t,y):
    return [y[2],y[3],1/((y[1]-y[0])**2),-1/((y[1]-y[0])**2)]
def y14(x):
    sim_output = numerical_solutions(F_num=F14_num, t0=-1, final_time_forward=1,y0=(1,2,5,10)).numerical_integrate()
    if x.dim() == 0:
        return torch.from_numpy(sim_output)
    else:
        return torch.from_numpy(
            numerical_solutions(F_num=F14_num, t0=0, final_time_forward=x[-1], dt=x[-1] / len(x),
                                y0=sim_output.tolist()).numerical_integrate())#True solution for F11
F14_plot_labels = ['q_1','q_2','p_1','p_2']

def F15(x,y):
    try:
       return torch.vstack([y[:,3],y[:,4],y[:,5],1/((y[:,1]-y[:,0])**2) + 1/((y[:,2]-y[:,0])**2),
                            -1/((y[:,1]-y[:,0])**2) + 1/((y[:,2]-y[:,1])**2),
                            -1/((y[:,2]-y[:,0])**2) - 1/((y[:,2]-y[:,1])**2)]).T
    except:
        return torch.hstack([y[3],y[4],y[5],1/((y[1]-y[0])**2) + 1/((y[2]-y[0])**2),
                             -1/((y[1]-y[0])**2) + 1/((y[2]-y[1])**2),
                             -1/((y[2]-y[0])**2) - 1/((y[2]-y[1])**2)])
def F15_num(y,t):
    return [y[3],y[4],y[5],1/((y[1]-y[0])**2) + 1/((y[2]-y[0])**2),
                             -1/((y[1]-y[0])**2) + 1/((y[2]-y[1])**2),
                             -1/((y[2]-y[0])**2) - 1/((y[2]-y[1])**2)]
# def y15(x):
#     sim_output = numerical_solutions(F_num=F15_num, t0=-1, final_time_forward=1,y0=(1,2,3,5,10,15)).numerical_integrate()
#     if x.dim() == 0:
#         return torch.from_numpy(sim_output)
#     else:
#         return torch.from_numpy(
#             numerical_solutions(F_num=F15_num, t0=0, final_time_forward=x[-1], dt=x[-1] / len(x),
#                                 y0=sim_output.tolist()).numerical_integrate())#True solution for F11
def y15(x):
    # sim_output = numerical_solutions(F_num=F15_num, t0=-1, final_time_forward=1,y0=(1,2,3,5,10,15)).numerical_integrate()
    if x.dim() == 0:
        return torch.from_numpy(np.array([1,2,3,5,10,15]))
    else:
        return torch.from_numpy(odeint(F15_num,[1,2,3,5,10,15],x.flatten()))#True solution for F15
F15_plot_labels = ['q_1','q_2','q_3','p_1','p_2','p_3']

# def F16(x,y):
#     k1 = 0.04
#     k2 = 3*1e7
#     k3 = 1e4
#     try:
#         return torch.vstack([-k1*y[:,0] + k3*y[:,1]*y[:,2],
#                              k1*y[:,0] - k2*y[:,1]**2 - k3 * y[:,1]*y[:,2],
#                              k2*y[:,1]**2]).T
#     except:
#         return torch.hstack([-k1*y[0] + k3*y[1]*y[2], k1*y[0] - k2*y[1]**2 - k3 * y[1]*y[2],k2*y[1]**2])
#
#
# def F16_num(t, y):
#     k1 = 0.04
#     k2 = 3*1e7
#     k3 = 1e4
#     return [-k1*y[0] + k3*y[1]*y[2], k1*y[0] - k2*y[1]**2 - k3*y[1]*y[2],k2*y[1]**2]
#
# def y16(x):
#     sim_output = numerical_solutions(F_num=F16_num, t0=-1, final_time_forward=1,y0=(1,0,0),solver='vode').numerical_integrate()
#     if x.dim() == 0:
#         return torch.from_numpy(sim_output)
#     else:
#         return torch.from_numpy(
#             numerical_solutions(F_num=F16_num, t0=0, final_time_forward=x[-1], dt=x[-1] / len(x),
#                                 y0=sim_output.tolist(),solver='vode').numerical_integrate())#True solution for F11
# F16_plot_labels = ['u_1','u_2']

def F19(x,y):
    k1 = 0.04
    k2 = 3*1e7
    k3 = 1e4
    try:
        return torch.vstack([-k1*y[:,0].flatten() + k3*y[:,1].flatten()*y[:,2].flatten(),
                             k1*y[:,0].flatten() - k2*y[:,1].flatten()**2 - k3*y[:,1].flatten()*y[:,2].flatten(),
                             k2*y[:,1].flatten()**2]).T
    except:
        return torch.hstack([-k1*y[0].flatten() + k3*y[1].flatten()*y[2].flatten(),
                             k1*y[0].flatten() - k2*y[1].flatten()**2 - k3 * y[1].flatten()*y[2].flatten(),
                             k2*y[1].flatten()**2])


def F19_num(y,t):
    k1 = 0.04
    k2 = 3*1e7
    k3 = 1e4
    return [-k1*y[0] + k3*y[1]*y[2], k1*y[0] - k2*y[1]**2 - k3*y[1]*y[2],k2*y[1]**2]

# def y19(x):
#     sim_output = numerical_solutions(F_num=F19_num, t0=-1, final_time_forward=0,y0=(1,0,0),solver='vode').numerical_integrate()
#     if x.dim() == 0:
#         return torch.from_numpy(sim_output)
#     else:
#         return torch.from_numpy(
#             numerical_solutions(F_num=F19_num, t0=0, final_time_forward=x[-1], dt=x[-1] / len(x),
#                                 y0=sim_output.tolist(),solver='vode').numerical_integrate())#True solution for F11
def y19(x):
    # sim_output = numerical_solutions(F_num=F19_num, t0=-1, final_time_forward=0,y0=(1,0,0),solver='vode').numerical_integrate()
    if x.dim() == 0:
        return torch.from_numpy(np.array([1,0,0]))
    else:
        return torch.from_numpy(odeint(F19_num,[1,0,0],x.flatten()))#True solution for F11
F19_plot_labels = ['u_1','u_2','u_3']

def F20(x,y):
    try:
       return torch.vstack([y[:,1],y[:,0]*0]).T
    except:
        return torch.hstack([y[1],y[0]*0])
def y20(x):
    return torch.hstack([x*0,x*0])

def F22(x,y):
    b = 0.05
    g = 9.81
    l = 1
    m = 1
    try:
        return torch.vstack([y[:,1].flatten(),-(b/m)*y[:,1].flatten()-(g/l)*torch.sin(y[:,0]).flatten()]).T
    except:
        return torch.hstack([y[1].flatten(),-(b/m)*y[1].flatten()-(g/l)*torch.sin(y[0]).flatten()])
def F22_num(y,t):
    b = 0.05
    g = 9.81
    l = 1
    m = 1
    q = y[0]
    p = y[1]
    return [p, - (b/m)*p - (g/l)*np.sin(q)]
def y22(x):
    # sim_output = odeint(F22_num,[1,1],torch.linspace(0,10,100))
    if x.dim() == 0:
        return torch.from_numpy(np.array([1,1]))#sim_output[0,:])
    else:
        return torch.from_numpy(odeint(F22_num,[1,1],x.flatten()))#True solution for F11
# def y22(x):
#     # sim_output = numerical_solutions(F_num=F22_num, t0=0, final_time_forward=0,y0=(1,0,0),solver='vode').numerical_integrate()
#     if x.dim() == 0:
#         return torch.from_numpy(np.array([1,1]))
#     else:
#         return torch.from_numpy(
#             numerical_solutions(F_num=F22_num, t0=0, final_time_forward=x[-1], dt=x[-1] / len(x),
#                                 y0=[1,1],solver='vode').numerical_integrate())#True solution for F11
F22_plot_labels = ['q','p']


def F26(x,y):
    try:
       return torch.vstack([y[:,1],y[:,0]*0 + 1]).T
    except:
        return torch.hstack([y[1],y[0]*0 + 1])
def y26(x):
    return torch.hstack([1/2*x**2,x]) #True solution for F26
F26_plot_labels = ['x','x_prime']

zero = torch.tensor(0)
number_reference = '19'
name = 'F' + number_reference
model_name = name + '_model.pt'
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

# if int(number_reference) > 10:
#     numerical=True
# else:
#     numerical=False

output = ode(F,y(zero),0.03125,epochs=int(1e4),numerical=numerical,plot_labels=plot_labels,batch_size=int(1e3))
             # , rule='simpson')
output.train(model_name)
output.plot(y,model_name,name)