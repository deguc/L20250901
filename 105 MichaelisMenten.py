#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def michaelis_menten(var,t,*args):

    S,ES,P = var
    kon,koff,k,E0 = args

    E = E0 -ES

    dS = koff * ES - kon * E * S
    dES = kon * E * S - koff * ES - k * ES
    dP = k * ES

    return [dS,dES,dP]

S0,E0 = 1000,1
kon,koff,k = 1,5,0.1

init = (S0,0,0)

t = np.linspace(0,13000,1000)

sol = odeint(michaelis_menten,init,t,args=(kon,koff,k,E0))

S = sol[:,0]
ES = sol[:,1]
P = sol[:,2]

fig = plt.figure(figsize=(6,4))
plt.plot(t,ES/E0,color='red',label='[ES]/[E0]')
plt.plot(t,P/S0,color='black',label='[P]/[S0]')
plt.plot(t,(S+ES)/S0,color='blue',label='([S]+[ES]/[S0]')
plt.title('Michaelis-Menten')
plt.xlabel('t')
plt.ylabel('Concentration')
plt.xticks([0,5000,10000])
plt.legend(bbox_to_anchor=(1,1))
plt.show()
