import numpy as np
from numba import njit
from numpy.random import rand

@njit()
def rk4(x,hesitancy, dxdt, dt, params):
    k1 = dxdt(x,hesitancy, params)
    k2 = dxdt(x + k1 / 2 * dt,hesitancy, params)
    k3 = dxdt(x + k2 / 2 * dt,hesitancy, params)
    k4 = dxdt(x + dt * k3,hesitancy, params)

    xnext = x + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return xnext

@njit()
def run_model(x, T, hesitancy,dxdt, params, dt, discard_len,int_steps):
    output=np.zeros((T+discard_len,x.size))
    output[0]=x
    for i in range(T+discard_len):
        output[i+1] = run_step(output[i],hesitancy,dxdt,dt,int_steps,params)
        hesitancy = hesitancy.mcmv(output[i+1,6],output[i+1,7],output[i+1,2],output[i+1,3]) #D_n,D_v,I_n,I_v
    return output[discard_len:]


@njit()
def run_step(x,hesitancy,dxdt,dt,int_steps,params):
    for i in range(int_steps):
        x = rk4(x,hesitancy, dxdt, dt, params)
    return x

@njit()
def SIR_dxdt(x, hesitancy,params):
    return np.array([-params[2]*x[0]*(x[2]+x[3])-hesitancy.global_hesitancy()*params[5]*x[0],
                     -params[1]*x[1]*(x[2]+x[3])+hesitancy.global_hesitancy()*params[5]*x[0],
                     params[2]*x[0]*(x[2]+x[3])-params[0]*x[2],
                     params[1]*x[1]*(x[2]+x[3])-params[0]*x[3],
                     params[0]*(1-params[4])*x[2],
                     params[0]*(1-params[3])*x[3],
                     params[0]*params[4]*x[2],
                     params[0]*params[3]*x[3]])


@njit()
def mcmove(config,c,beta,N,D_n,D_v,I_n,I_v):
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = config[a, b]
            nb = config[(a + 1) % N, b] + config[a, (b + 1) % N] + config[(a - 1) % N, b] + config[a, (b - 1) % N]
            cost = 2 * s * nb - 2 * s *(c * I_v/(I_n+I_v+1e-12) + (1-c)* D_v/(D_n+D_v+1e-12)) + 2 * s *(c * I_n/(I_n+I_v+1e-12) + (1-c)* D_n/(D_n+D_v+1e-12))
            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost * beta):
                s *= -1
            config[a, b] = s
    return config

@njit()
def run_mcmove(config,T,discard_len,c,beta,N,D_n,D_v,I_n,I_v):
    for i in range(discard_len):
        config = mcmove(config,c,beta,N,D_n,D_v,I_n,I_v)
    output=np.zeros((T,N,N))
    for i in range(T):
        output[i]=mcmove(config,c,beta,N,D_n,D_v,I_n,I_v)
    return output


