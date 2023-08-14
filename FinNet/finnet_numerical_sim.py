import numpy as np
from scipy.integrate import ode

class numerical_solutions(object):

    def __init__(self,F_num,t0,final_time_forward,y0=(-1,-1,-1,-1),dt=0.001,solver='dop853'):
        self.F_num = F_num
        self.t0 = t0
        self.final_time_forward = final_time_forward
        self.y0 = y0
        self.dt = dt
        self.size = int(np.round((self.final_time_forward/self.dt),0))
        self.solver=solver

    def numerical_integrate(self):

        solutions = np.zeros((self.size +2,len(self.y0)))
        steps = np.zeros((self.size+2,1))
        solutions[0] = self.y0
        steps[0] = self.t0
        dynamical_system = ode(self.F_num)
        dynamical_system.set_integrator(self.solver)
        dynamical_system.set_initial_value(self.y0, self.t0)
        i = 1
        while dynamical_system.successful() and dynamical_system.t < self.final_time_forward:
            dynamical_system.integrate(dynamical_system.t + self.dt)
            solutions[i] = dynamical_system.y
            steps[i] = dynamical_system.t
            if (i >= self.final_time_forward / self.dt):
                break;
            i += 1
        
        if self.t0 == -1:
            return dynamical_system.y
        else:
            return solutions[1:-1,:]

# def Fs(t ,y):
# 
#     return [y[2],y[3],-y[0]/(y[0]**2+y[1]**2)**(3/2),-y[1]/(y[0]**2+y[1]**2)**(3/2)]
# 
# Niters = 1
# Dt = 0.001
# y0=(-1,-1,-1,-1)
# t0=-1
# size = Niters/Dt
# size = int(size)
# solRk4 = np.zeros((size +2,len(y0)))
# solRk4[0] = y0
# 
# Q = []
# Q.append(y0[0])
# P = []
# P.append(y0[1])
# t = []
# t.append(t0)
# 
# 
# Orbit = ode(Fs)
# Orbit.set_integrator('dop853')
# Orbit.set_initial_value(y0, t0)
# i = 1
# while Orbit.successful() and Orbit.t < Niters:
#     Orbit.integrate(Orbit.t + Dt)
#     solRk4[i] = Orbit.y
#     t.append(Orbit.t)
#     print(Orbit.t, Orbit.y)
#     # print(Orbit.successful())
#     # q,p = Orbit.y
#     # Q.append(q)
#     # P.append(p)
#     # if i > 2 and np.abs(Q1[-1]**2+Q2[-1]**2 - 1) < 10**(-4):
#     #     break
#     if( i >= Niters/Dt):
#         break;
#     i += 1
# print(Orbit.y)