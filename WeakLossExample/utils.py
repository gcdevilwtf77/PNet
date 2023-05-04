import torch

def y(model,x,initial_condition, y_prime):
    # return x + (1/2)*model(x)*x**2
    # return 1 - x + (1/2)*model(x)*x**2
    return initial_condition + y_prime*x + (1/2)*model(x)*x**2
def F(x,y):
    # return -y + torch.cos(x) + torch.sin(x)
    return  -y + x
def integrate(model,x,dx,rule,initial_condition, y_prime):
    x_left = x - dx/2
    x_right = x + dx/2
    y_left, y_mid, y_right = y(model,x_left,initial_condition, y_prime), y(model,x,initial_condition, y_prime), \
                            y(model,x_right,initial_condition, y_prime)
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

def loss_train(model,x,dx,initial_condition,y_prime,rule):
    return dx*torch.sum(torch.abs(y(model,x+dx/2,initial_condition,y_prime) - initial_condition \
                           - integrate(model,x,dx,rule,initial_condition, y_prime)))
def train(model,x,dx,initial_condition,y_prime,rule,epochs,optimizer,scheduler):
    model.train()
    for i in range(epochs):

        # Construct loss function and compute gradients and optimizer updates
        # We are solving the ODE y'(x) = -y(x) + cos(x) + sin(x), y(0)=0
        # whose solution is y(x)=sin(x). We're learning a function
        # y(x)=y(0) + y'(0)x + (1/2)f(x)x^2,
        # where f(x) is the neural network.

        optimizer.zero_grad()
        loss = loss_train(model,x,dx,initial_condition,y_prime,rule)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 1000 == 0:
            print(i, loss.item())

    torch.save(model, 'weak_loss_model.pt')