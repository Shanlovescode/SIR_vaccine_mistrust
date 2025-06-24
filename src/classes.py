from model_functions import *
import numpy as np
from numba import njit

class SIR_vaccination():
    def __init__(self, alpha,beta_v,beta_n,p_v,p_n,v0,dt=0.01, int_steps=10, ic=np.array([]), ic_seed=0):
        self.params = np.array([alpha,beta_v,beta_n,p_v,p_n,v0])
        self.dxdt = SIR_dxdt
        self.hesitancy=hesitancy
        self.dt = dt
        self.int_steps = int_steps
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(8) * 2 - 1) * np.array([0.5, 0.5, 0.,0.,0.,0.,0.,0.])
        elif ic.size == 8:
            self.state = ic.flatten()
        else:
            raise ValueError

    def run(self, T,discard_len): #certain length is discarded for spin up
        output = run_model(self.state, T, self.hesitancy, self.dxdt, self.params,self.dt,discard_len,self.int_steps)
        self.state = output[-1]
        return output

class hesitancy():#Ising model hesitancy. The hesitancy variable is binary over 1 and -1
    def __init__(self, c,N,beta,seed=0):
        self.N=N
        self.c=c
        self.beta=beta
        self.config = 2*np.random.randint(2, size=(N,N))-1
    def mcmv(self,D_n,D_v,I_n,I_v):
        output=mcmove(self.config,self.c,self.beta,self.N,D_n,D_v,I_n,I_v)
        self.config=output
        return output
    def global_hesitancy(self):
        config=self.config
        mag = np.sum(config[config>0])/(self.N*self.N)
        return mag
    def run_mcmv(self, T,discard_len,D_n,D_v,I_n,I_v): #test mcmv!\
        print(self.config)
        output = run_mcmove(self.config, T, discard_len,self.c,self.beta,self.N,D_n,D_v,I_n,I_v)
        self.config = output[-1]
        return output








