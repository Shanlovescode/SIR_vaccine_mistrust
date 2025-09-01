import numpy as np
from src.classes import hesitancy,SIR_vaccination
import matplotlib.pyplot as plt
import seaborn as sns
c_array=1.0-np.array([0,0.2,0.4,0.6,0.8,1.0])
deathpeak=np.zeros(len(c_array))
hesitancy_array=np.zeros(len(c_array))
for i in range(len(c_array)):
    hesitancy_Ising=hesitancy(c=c_array[i],N=100,beta=0.25,test_SIR=False);
    initial_conditions=np.array([0.499,.499,0.002,0,0,0,0,0]) #Sn,Sv,In,Iv,Rn,Rv,Dn,Dv
    SIR=SIR_vaccination(hesitancy=hesitancy_Ising,alpha=0.1,beta_v=0.3,beta_n=0.3,p_v=0.001,p_n=0.01,v0=np.log(0.01)/(-365),ic=initial_conditions);
    output=SIR.run(T=52,discard_len=0);
    Sn=output[:,0];
    Sv=output[:,1];
    In=output[:,2];
    Iv=output[:,3];
    Rn=output[:,4];
    Rv=output[:,5];
    Dn=output[:,6];
    Dv=output[:,7];
    print((Dn+Dv)[-1])
    deathpeak[i]=(Dv+Dn)[-1]
    hesitancy_array[i]=hesitancy_Ising.global_hesitancy()
print(1-hesitancy_array)
plt.figure(figsize=(6, 6))
plt.plot(1-c_array,(deathpeak[0]-deathpeak)/deathpeak[0],linewidth=2.0)
plt.ylim([0,0.08])
plt.xlabel('Degree of death focused communication',fontsize = 15)
plt.ylabel('Proportion of deaths avoided relative to baseline', fontsize = 15)
#plt.figure(2)
#plt.plot(1-c_array,hesitancy_array)
#plt.ylabel('trust after epidemic ')
#plt.xlabel('Degree of death focused communication (1-c)')
plt.show()