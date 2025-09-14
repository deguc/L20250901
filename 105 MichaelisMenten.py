#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def michaelis_menten(t,var,*args):

    S,ES,P = var
    kon,koff,k,E0 = args

    E = E0 -ES

    dS = koff * ES - kon * E * S
    dES = kon * E * S - koff * ES - k * ES
    dP = k * ES

    return [dS,dES,dP]

S0,E0 = 1000,1
kon,koff,k = 1,5,0.1

init = np.array([S0,0,0])

t_span=(0,15000)
t_eval = np.linspace(0,13000,1000)

sol = solve_ivp(
    michaelis_menten,
    t_span,
    init,
    args=(kon,koff,k,E0),
    method='LSODA',
    t_eval = t_eval,
    rtol=5e-4,
    atol=1e-6
)

t = sol.t
S,ES,P = sol.y

fig = plt.figure(figsize=(6,4))
plt.plot(t,ES/E0,color='red',label='[ES]/E0')
plt.plot(t,P/S0,color='black',label='[P]/S0')
plt.plot(t,(S+ES)/S0,color='blue',label='([S]+[ES])/S0')
plt.title('Michaelis-Menten')
plt.xlabel('t')
plt.ylabel('Concentration')
plt.xticks([0,5000,10000])
plt.legend(bbox_to_anchor=(1,1))
plt.show()
