import numpy as np
from src.classes import hesitancy
import matplotlib.pyplot as plt
import seaborn as sns

c=1;
#temp=0.4; #ferromagnetic regime
temp=3; #paramagnetic regime
N=64;
hesitancy_Ising=hesitancy(c=c,beta=1/temp,N=N);
D_n=1.; D_v=1.;I_n=1.;I_v=1.; #same as Ising model
D_n=5.; D_v=1.;I_n=5.;I_v=1.; #reduce hesitancy
D_n=1.; D_v=5.;I_n=1.;I_v=5.; #increase hesitancy
output=hesitancy_Ising.run_mcmv(2001,discard_len=0,D_n=D_n,D_v=D_v,I_n=I_n,I_v=I_v)
plt.rcParams.update({'font.size': 22})
fig, axes = plt.subplots(2, 3, figsize=(30,12))
axes=axes.flatten()
plot_ind=np.array([0,1,4,32,100,2000])
for i in range(len(axes)):
    sns.heatmap(output[plot_ind[i]],ax=axes[i],vmin=-1,vmax=1)
    axes[i].axis('off')
    axes[i].set_title('time='+str(plot_ind[i]))
#fig.suptitle('Ferromagnetic')
#fig.suptitle('paramagnetic')
#fig.suptitle('paramagnetic+I_n>I_v,c=1')
fig.suptitle('paramagnetic+I_n<I_v,c=1')
plt.show()