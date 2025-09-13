#%%
import numpy as np
import matplotlib.pyplot as plt

def onehot(x):

    k = x.max()+1
    
    return np.identity(k)[x]


def Dataset(x,y,size=10,scale=0.1):

    X,Y = [],[]
    k = x.shape[1]

    for x0,y0 in zip(x,y):

        X += [x0+np.random.rand(size,k)*scale]
        Y += [np.full(size,y0)]
    
    X = np.vstack(X)
    Y = onehot(np.hstack(Y))

    return X,Y


class DataLoader:

    def __init__(self,dataset,batch_size):
        
        x,y = dataset
        self.data_size = x.shape[0]
        idx = np.random.permutation(self.data_size)
        self.x,self.y = x[idx],y[idx]
        self.batch_size = batch_size
        self.count = 0
    
    def __len__(self):
        return self.data_size // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        i = self.count
        j = self.count + self.batch_size
        
        if j > self.data_size:

            self.count = 0
            idx = np.random.permutation(self.data_size)
            self.x,self.y = self.x[idx],self.y[idx]
            
            raise StopIteration
        
        else:
            self.count += self.batch_size
            return self.x[i:j],self.y[i:j]


def zeros_ps(ps):

    gs = []

    for p in ps:

        gs += [np.zeros_like(p)]
    
    return gs


class Linear:

    def __init__(self,d_in,d_out):

        self.ps = [
            np.random.rand(d_in,d_out),
            np.zeros(d_out)
        ]
        self.gs = zeros_ps(self.ps)

        self.inputs = None
    
    def __call__(self,x):

        self.inputs = x

        return x @ self.ps[0] + self.ps[1]

    def backward(self,dout):

        self.gs[0][...] = self.inputs.T @ dout
        self.gs[1][...] = np.sum(dout,axis=0)

        return dout @ self.ps[0].T
    

class Sigmoid:

    def __init__(self):

        self.ps,self.gs = [],[]
        self.out = None
    
    def __call__(self,x):
        
        out = 1 / (1+np.exp(-np.clip(x,-500,500)))
        self.out = out

        return out
    
    def backward(self,dout):

        return dout * self.out * (1-self.out)


class ReLU:

    def __init__(self):
        
        self.ps,self.gs = [],[]
        self.mask = None
    
    def __call__(self,x):

        self.mask = (x < 0)
        x[self.mask] = 0

        return x
    
    def backward(self,dout):

        dout[self.mask] = 0

        return dout

class LeakyReLU:

    def __init__(self,rate=0.01):
        
        self.ps,self.gs = [],[]
        self.mask = None
        self.rate = rate
    
    def __call__(self,x):

        self.mask = (x < 0)
        x[self.mask] *= self.rate

        return x
    
    def backward(self,dout):

        dout[self.mask] *= self.rate

        return dout


class SGD:

    def __init__(self,lr=0.1):
        
        self.lr = lr
    
    def __call__(self,ps,gs):

        for p,g in zip(ps,gs):

            p -= self.lr * g


class Layers:

    def __init__(self,layers):

        self.layers = layers
        self.ps,self.gs = [],[]

        for l in layers:

            self.ps += l.ps
            self.gs += l.gs
    
    def __call__(self,x):

        for l in self.layers:
            x = l(x)
        
        return x

    def backward(self,dout):

        for l in reversed(self.layers):
            dout = l.backward(dout)
    
    def pred(self,x):
        return np.argmax(self(x),axis=-1)

    def fit(self,data,epochs=-1,lr=0.1):
        
        optimizer = SGD(lr=lr)
        loss = []
        iter = len(data)

        for _ in range(epochs):

            l = 0

            for x,t in data:

                y = self(x)
                l_,dout = Softmax(y,t)
                l += l_
                self.backward(dout)
                optimizer(self.ps,self.gs)
        
            loss += [l/iter]

        return loss


def Softmax(y,t):

    
    c = np.max(y,axis=-1,keepdims=True)
    e = np.exp(y-c)
    z = np.sum(e,axis=-1,keepdims=True)
    out = e/z
    
    eps = 1e-6
    loss = -np.sum(t*np.log(out+eps)) / y.shape[0]

    dout = out - t

    return loss,dout

def Model(data,layers,epochs=100,lr=0.1,batch_size=10):
    

    model = Layers(layers)
    loss = model.fit(data,epochs=epochs,lr=lr)
    return loss    

np.set_printoptions(precision=2,suppress=True)

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

dataset = Dataset(x,y,size=10,scale=0.1)
x_train,y_train = dataset
data = DataLoader(dataset=dataset,batch_size=10)

d_in = x.shape[1]
d_h = 4*d_in
d_out = y.max()+1

layers = [
    Linear(d_in,d_h),
    Sigmoid(),
    Linear(d_h,d_out)
]
model = Layers(layers)
loss = model.fit(data,epochs=30,lr=0.1)
pred = model.pred(x)

epochs=50
lr =0.1
batch_size=10

layers1 = [
    Linear(d_in,d_h),
    Sigmoid(),
    Linear(d_h,d_out)
]
layers2 = [
    Linear(d_in,d_h),
    ReLU(),
    Linear(d_h,d_out)
]

layers3 = [
    Linear(d_in,d_h),
    LeakyReLU(),
    Linear(d_h,d_out)
]
loss1=Model(data,layers1,epochs=epochs,lr=lr,batch_size=batch_size)
loss2=Model(data,layers2,epochs=epochs,lr=lr,batch_size=batch_size)
loss3=Model(data,layers3,epochs=epochs,lr=lr,batch_size=batch_size)

fig,ax = plt.subplots()
plt.title('Loss Fuction')
plt.xlabel('epochs')
plt.ylabel('cross entropy')
ax.plot(loss1,color='blue',label='Sigmoid')
ax.plot(loss2,color='red',label='ReLU')
ax.plot(loss3,color='black',label='LeakyReLU')

ax.legend()
plt.show()

print(pred)
