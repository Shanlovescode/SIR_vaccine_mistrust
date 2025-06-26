from src.model_functions import *
import numpy as np
from numba import njit

class SIR_vaccination():
    def __init__(self, hesitancy,alpha,beta_v,beta_n,p_v,p_n,v0,dt=0.01, int_steps=10, ic=np.array([]), ic_seed=0):
        self.params = np.array([alpha,beta_v,beta_n,p_v,p_n,v0])
        self.dxdt = SIR_dxdt
        self.hesitancy=hesitancy
        self.dt = dt
        self.int_steps = int_steps
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(8) * 2 - 1) * np.array([1., 1., 1.,1.,1.,1.,1.,1.])
        elif ic.size == 8:
            self.state = ic.flatten()
        else:
            raise ValueError

    def run(self, T,discard_len): #certain length is discarded for spin up #change this part!!!! don't pass classes to run_step
        output = np.zeros((T + discard_len+1, self.state.size))
        output[0] = self.state
        print(output[0])
        for i in range(T + discard_len):
            output[i + 1] = run_step(output[i], self.hesitancy.global_hesitancy(), self.dxdt, self.dt, self.int_steps, self.params)
            self.hesitancy.run_mcmv(T=self.int_steps,discard_len=0,D_n=output[i + 1, 6], D_v=output[i + 1, 7], I_n=output[i + 1, 2],
                                       I_v=output[i + 1, 3])  # D_n,D_v,I_n,I_v
        return output

class hesitancy():#Ising model hesitancy. The hesitancy variable is binary over 1 and -1
    def __init__(self, c,N,beta,seed=0,test_SIR=False):
        self.N=N
        self.c=c
        self.beta=beta
        self.config = 2*np.random.randint(2, size=(N,N))-1
        self.test_SIR=test_SIR
    def mcmv(self,D_n,D_v,I_n,I_v):
        output=mcmove(self.config,self.c,self.beta,self.N,D_n,D_v,I_n,I_v)
        self.config=output
        return output
    def global_hesitancy(self):
        if self.test_SIR:
            return 0.5
        config=self.config
        mag = np.sum(config[config>0])/(self.N*self.N)
        return mag
    def run_mcmv(self, T,discard_len,D_n,D_v,I_n,I_v): #test mcmv!
        output = run_mcmove(self.config, T, discard_len,self.c,self.beta,self.N,D_n,D_v,I_n,I_v)
        self.config = output[-1]
        return output








