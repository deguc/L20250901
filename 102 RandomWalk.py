import numpy as np
import matplotlib.pyplot as plt

n = 100
t = 1000
x = np.zeros((t,n))
k = np.arange(0,t,t/10,dtype=np.int64)

for i in range(1,t):
    x[i] = x[i-1]+np.random.choice([-1,1],n)

msd = np.mean(x[k,:]**2,axis=1)
print(msd)

fig = plt.figure(figsize=[6,8])

ax1 = fig.add_subplot(211)
ax1.set_title('Random Walk')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.plot(x,color='blue',alpha=0.2)

ax2 = fig.add_subplot(212)
ax2.set_title('MSD')
ax2.set_xlabel('t')
ax2.set_ylabel('MSD')
ax2.scatter(k,msd,color='orange')
ax2.plot(k,k,c='red',alpha=0.4)
plt.tight_layout()
plt.show()



