#%%
import numpy as np
import matplotlib.pyplot as plt

def onehot(x):
    
    k = x.max()+1

    return np.identity(k)[x]


def DataSet(x,y,size=10,scale=0.1):

    X,Y = [],[]
    k = x.shape[1]

    for x0,y0 in zip(x,y):
        
        X += [x0+np.random.randn(size,k)*scale]
        Y += [np.full(size,y0)]
    
    X = np.vstack(X)
    Y = onehot(np.hstack(Y))

    return (X,Y)


class DataLoader:

    def __init__(self,dataset,batch_size=10):

        self.x,self.y = dataset
        self.data_size = self.x.shape[0]
        self.cnt = 0
        self.batch_size= batch_size
        self.shuffle()
    
    def shuffle(self):
        
        idx = np.random.permutation(self.data_size)
        self.x = self.x[idx]
        self.y = self.y[idx]

    def __len__(self):
        return self.data_size // self.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        i = self.cnt
        j = i + self.batch_size

        if j > self.data_size:
            self.cnt = 0
            self.shuffle()
            raise StopIteration
        else:
            self.cnt += self.batch_size
            return self.x[i:j],self.y[i:j]
    
    def get_dim(self):

        d_in = self.x.shape[1]
        d_h = 4*d_in
        d_out = self.y.shape[1]

        return d_in,d_h,d_out

            
def zeros_ps(ps):
    
    gs = []

    for p in ps:
        gs += [np.zeros_like(p)]
    
    return gs

class Linear:

    def __init__(self,d_in,d_out):
        
        std = np.sqrt(d_in/2)

        self.ps = [
            np.random.randn(d_in,d_out)/std,
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


class ReLU:

    def __init__(self):
        
        self.ps,self.gs = [],[]
        self.mask = None
    
    def __call__(self,x):

        self.mask = x < 0
        x[self.mask] = 0

        return x
    
    def backward(self,dout):

        dout[self.mask] = 0

        return dout


class SGD:

    def __init__(self,lr=0.1):
        self.lr = lr
    
    def __call__(self,ps,gs):

        for p,g in zip(ps,gs):
            p -= self.lr * g

class Momentum:

    def __init__(self,lr=0.1,alpha=0.90):
        self.lr = lr
        self.ms = []

        self.alpha = alpha

    def __call__(self,ps,gs):
        
        if self.ms == []:
            self.ms = zeros_ps(ps)

        for p,g,m in zip(ps,gs,self.ms):
            m = self.alpha * m - self.lr*g
            p += m


class Adagrad:

    def __init__(self,lr=0.01):

        self.lr = lr
        self.hs = []
    
    def __call__(self,ps,gs):
        eps = 1e-7

        if self.hs == []:
            self.hs = zeros_ps(ps)

        for p,g,h in zip(ps,gs,self.hs):

            h += g*g

            p -= self.lr * g/(np.sqrt(h)+eps)

class RMSprop:

    def __init__(self,lr=0.01,beta=0.9):

        self.lr = lr
        self.hs = []
        self.beta = beta
    
    def __call__(self,ps,gs):
        eps = 1e-7

        if self.hs == []:
            self.hs = zeros_ps(ps)

        for p,g,h in zip(ps,gs,self.hs):

            h = self.beta * h + (1-self.beta)*g*g

            p -= self.lr * g/(np.sqrt(h)+eps)

class Adam:

    def __init__(self,lr=0.01,alpha=0.25,beta=0.9):

        self.lr = lr
        self.hs = []
        self.ms = []
        self.alpha= alpha
        self.beta = beta
        self.n = 0
    
    def __call__(self,ps,gs):
        eps = 1e-7
        self.n += 1

        if self.hs == []:
            self.hs = zeros_ps(ps)
            self.ms = zeros_ps(ps)

        for p,g,h,m in zip(ps,gs,self.hs,self.ms):

            h = self.beta * h + (1-self.beta)*g*g
            m = self.alpha * m +(1-self.alpha)*g

            h0 =  h/(1-self.beta**self.n)
            m0 = m/(1-self.alpha**self.n)

            p -= self.lr * m0/(np.sqrt(h0)+eps)


def Softmax(y,t):

    eps = 1e-6
    N = y.shape[0]

   
    c = np.max(y,axis=1,keepdims=True)
    e = np.exp(y-c)
    z = np.sum(e,axis=1,keepdims=True)
    out = e/z

    loss = -np.sum(t*np.log(out+eps)) / N

    dout = (out-t) / N

    return loss,dout


class Layers:

    def __init__(self,layers):

        self.layers = layers
        self.ps,self.gs = [],[]

        for l in self.layers:
            self.ps += l.ps
            self.gs += l.gs
    
    def __call__(self,x):

        for l in self.layers:
            x = l(x)
        
        return x

    def backward(self,dout):

        for l in reversed(self.layers):
            dout = l.backward(dout)
        
        return dout

    def fit(self,data,epochs=100,optimizer=SGD()):

        loss = []

        for _ in range(epochs):

            l = 0

            for x,t in data:
                y = self(x)
                l_,dout = Softmax(y,t)
                l += l_
                self.backward(dout)
                optimizer(self.ps,self.gs)

            loss += [l/len(data)]

        return loss
                    

    def pred(self,x):

        return np.argmax(self(x),axis=1)

def model(data,optimizer,epochs=100):
    

    d_in,d_h,d_out = data.get_dim()

    layers = [
        Linear(d_in,d_h),
        ReLU(),
        Linear(d_h,d_out)
    ]
    
    model = Layers(layers)
    loss = model.fit(data,epochs=epochs,optimizer=optimizer)
    pred = model.pred(x)

    print(pred)
    
    return loss


x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

epochs = 100
lr = 0.011

dataset = DataSet(x,y,size=10,scale=0.1)
data = DataLoader(dataset,batch_size=10)


loss1 = model(data,epochs=epochs,optimizer=SGD(lr=lr))
loss2 = model(data,epochs=epochs,optimizer=Momentum(lr=lr,alpha=0.2))
loss3 = model(data,epochs=epochs,optimizer=Adagrad(lr=lr))
loss4 = model(data,epochs=epochs,optimizer=RMSprop(lr=lr))
loss5 = model(data,epochs=epochs,optimizer=Adam(lr=lr))


plt.title('Loss Function')
plt.xlabel('epochs')
plt.ylabel('cross_entropy')
plt.plot(loss1,color='blue',label='SGD')
plt.plot(loss2,color='red',label='Momentum')
plt.plot(loss3,color='green',label='Adagrad')
plt.plot(loss4,color='orange',label='RMSprop')
plt.plot(loss5,color='black',label='RMSprop')
plt.legend()
plt.show()

