import numpy as np
from scipy.integrate import ode
from time import time

class numerical_solutions(object):

    def __init__(self,F_num,t0,final_time_forward,y0,dt=0.001,solver='dop853'):
        self.F_num = F_num
        self.t0 = t0
        self.final_time_forward = final_time_forward
        self.y0 = y0
        self.dt = dt
        self.size = int(np.abs(np.round((self.final_time_forward/self.dt),0)))
        self.solver=solver

    def numerical_integrate(self):

        solutions = np.zeros((self.size +2,len(self.y0)))
        steps = np.zeros((self.size+2,1))
        solutions[0] = self.y0
        steps[0] = self.t0
        dynamical_system = ode(self.F_num)
        if self.solver == 'vode':
            dynamical_system.set_integrator(self.solver, method='bdf')
        else:
            dynamical_system.set_integrator(self.solver)
        dynamical_system.set_initial_value(self.y0,self.t0)
        i = 1
        # start = time()
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
            # print('Num Solver time: ' + str(time() - start))
            # return solutions[1:-1:100,:]
            return solutions[0:-2,:]

