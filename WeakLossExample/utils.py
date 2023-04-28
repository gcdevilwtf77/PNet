import torch

def y(model,x):
    return x + (1/2)*model(x)*x**2

def F(x,y):
    return -y + torch.cos(x) + torch.sin(x)

def integrate(model,x,dx,rule):
    x_left = x - dx/2
    x_right = x + dx/2
    y_left, y_mid, y_right = y(model,x_left), y(model,x), y(model,x_right)
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
