import numpy as np
import matplotlib.pyplot as plt

def onehot(x):
    k=x.max()+1
    return np.identity(k)[x]

def softmax(x):
    c = np.max(x,axis=-1,keepdims=True)
    e = np.exp(x-c)
    z = np.sum(e,axis=-1,keepdims=True)
    return e/z

def cross_entropy(y,t):
    eps = 1e-6
    return -np.sum(t*np.log(y+eps))/y.shape[0]

def Dataset(x,y,size=10,scale=0.1):

    X,Y = [],[]
    k = x.shape[1]

    for x0,y0 in zip(x,y):
        X += [x0+np.random.normal(size=(size,k),loc=0,scale=scale)]
        Y += [np.full(size,y0)]
    
    X = np.vstack(X)
    Y = onehot(np.hstack(Y))

    return X,Y

def zeors_ps(ps):

    gs = []

    for p in ps:

        gs += [np.zeros_like(p)]
    
    return gs


class Linear:

    def __init__(self,d_in,d_out):

        self.ps = [
            np.random.randn(d_in,d_out),
            np.zeros(d_out)
        ]
        self.gs = zeors_ps(self.ps)
        self.inputs = None
    
    def __call__(self,x):
        
        self.inputs = x

        return x @ self.ps[0] + self.ps[1]

    def backward(self,dout):

        self.gs[0][...] = self.inputs.T @ dout
        self.gs[1][...] = np.sum(dout,axis=0)

        return dout @ self.ps[0].T

class ReLU:

    def __init__(self):
        self.ps,self.gs = [],[]
        self.mask = None
    def __call__(self,x):

        self.mask = (x<0)
        x[self.mask] = 0

        return x

    def backward(self,dout):
        dout[self.mask] = 0
        return dout


class Sigmoid:

    def __init__(self):

        self.ps,self.gs = [],[]
        self.out = None
    
    def __call__(self,x):

        y = 1/(1+np.exp(-np.clip(x,-500,500)))
        self.out = y

        return y
    
    def backward(self,dout):

        a = self.out

        return dout * a*(1-a)

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

        logits = self(x)
        
        return np.argmax(logits,axis=-1)

    def fit(self,x,t,epochs=100,batch_size=10,lr=0.1):

        loss = []
        optimizer = SGD(lr=lr)
        data_size = x.shape[0]
        iter = data_size // batch_size

        for __ in range(epochs):

            idx = np.random.choice(data_size,batch_size)
            x_,t_ = x[idx],t[idx]
            y =softmax(self(x_))
            loss += [cross_entropy(y,t_)]
            dout = (y- t_)/batch_size
            self.backward(dout)
            optimizer(self.ps,self.gs)
                
        
        return loss
        
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,2,3])

x_train,y_train = Dataset(x,y)

d_in = x.shape[1]
d_h = d_in*4
d_out = y_train.shape[1]

layers = [
    Linear(d_in,d_h),
    ReLU(),
    Linear(d_h,d_out)
]
model = Layers(layers)
loss=model.fit(x_train,y_train,epochs=1000,batch_size=10,lr=0.1)
pred = model.pred(x)
plt.plot(loss)
plt.show()
print(pred)
