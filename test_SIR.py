import numpy as np
from src.classes import hesitancy,SIR_vaccination
import matplotlib.pyplot as plt
import seaborn as sns

hesitancy_Ising=hesitancy(c=1,N=64,beta=0.25,test_SIR=False);
initial_conditions=np.array([0.5,0.5,0.1,0.1,0,0,0,0]) #Sn,Sv,In,Iv,Rn,Rv,Dn,Dv
SIR=SIR_vaccination(hesitancy=hesitancy_Ising,alpha=1/7,beta_v=0.3,beta_n=0.3,p_v=0.02,p_n=0.1,v0=np.log(0.01)/(-365),ic=initial_conditions);
output=SIR.run(T=300,discard_len=0);
Sn=output[:,0];
Sv=output[:,1];
In=output[:,2];
Iv=output[:,3];
Rn=output[:,4];
Rv=output[:,5];
Dn=output[:,6];
Dv=output[:,7];
plt.plot(Dn)
plt.plot(Dv)
print(hesitancy_Ising.global_hesitancy())
plt.show()