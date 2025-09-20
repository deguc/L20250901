import numpy as np
import matplotlib.pyplot as plt

def exp_free_energy(mu,J,N,M):
    eps = -1
    e = (eps-mu)*N+J*M

    return np.exp(-e)


def pauling(mu,J):

    eps = -1
    M = [0,0,1,3,6]
    E = np.array([((eps-mu)*i+J*M) for i,M in enumerate(M)])
    X = np.exp(-E)

    K = np.array([1,4,6,4,1])
    L = np.array([0,4,12,12,4])

    Z = K @ X
    N = (L @ X)/Z

    return N

p = np.logspace(-3,-0.7,base=10)
mu = np.log(p)

plt.title('AlLosteric Effect')
plt.xticks([0,0.1,0.2])
plt.yticks([0,2.0,4.0])
plt.xlabel(r'$p(o_2)/p_0$')
plt.ylabel('N')

for J in [-2,-1,0]:

    N = pauling(mu,J)
    plt.plot(p,N,label=f'{J:.1f}$k_B$T')

plt.legend()
plt.show()
