#%%
import numpy as np
import matplotlib.pyplot as plt

n = 10000
m = 100
k = 1000

x = np.random.rand(n)
s_x = np.random.choice(x,[k,m],replace=True)
s_mean = np.mean(s_x,axis=1)
mu = np.mean(x)
var = np.var(x,ddof=0)/m
std = np.sqrt(var)
t = np.linspace(mu-std*4,mu+std*4,100)
var2 = var*2.
pdf = np.exp(-(t-mu)**2/var2)/np.sqrt(np.pi*var2)

fig = plt.figure()
plt.title('Original data')
plt.xlabel('x')
plt.ylabel('density')
plt.hist(x,bins=100,density=True,color='blue')

fig = plt.figure()
plt.title('Central Limit Theorem')
plt.xlabel('mean')
plt.ylabel('density')
plt.hist(s_mean,bins=100,density=True,color='red',alpha=0.4)
plt.plot(t,pdf,color='red')
plt.show()
