import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz(t,var,*args):

    x,y,z = var
    s,r,b = args

    dx = s * (y-x)
    dy = x*(r-z)-y
    dz = x*y - b*z

    return (dx,dy,dz)

init = (0.1,0.1,0.1)
t_span = (0,100)
t_eval = np.linspace(0,40,4000)

s = 10
r = 28
b = 8/3

sol = solve_ivp(lorenz,t_span,init,args=(s,r,b),t_eval=t_eval)
t = sol.t
x,y,z = sol.y

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(x,y,z)


