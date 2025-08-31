#%%
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(precision=2,suppress=True)
np.random.seed(123)

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,0,1])

W = np.random.randn(2)
b = 0


epochs = 20
lr = 0.2

for i in range(epochs):

    out =  (x @ W + b > 0) * 1
    print(f'epochs={i}    logits={out}')

    dout = out-y

    dW = x.T @ dout
    db = np.sum(dout)

    W -= lr*dW
    b -= lr*db

#%%
def step(x):
    return (x>0) * 1

class Perceptron:

    def __init__(self,act=step):

        self.W = np.random.random(2)
        self.b = 0
        self.inputs = None
        self.lr = 0.5
        self.act = act

    def __call__(self,x):
        self.inputs = x
        y = x @ self.W + self.b

        return self.act(y)
    
    def backward(self,dout):

        dW = self.inputs.T @ dout
        db = np.sum(dout)

        self.W -= self.lr * dW
        self.b -= self.lr * db

    def fit(self,x,y,epochs=20):

        for i in range(epochs):
            
            dout = self(x) - y
            self.backward(dout)
            loss = np.sum(dout**2)
            print(f'epoch = {i}   loss = {loss}')

np.set_printoptions(precision=2,suppress=True)
np.random.seed(123)


x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

ppt = Perceptron(act=step)
ppt.lr = 0.1
ppt.fit(x,y,epochs=10)
pred = ppt(x)
print(f'\nlabels  {y}')
print(f'pred    {pred}')

#%%
def sigmoid(x):
    eps = 1e-6

    return 1/(1+np.exp(-x))

class Perceptron:

    def __init__(self,act=sigmoid):

        self.W = np.random.random(2)
        self.b = 0
        self.inputs = None
        self.lr = 0.1
        self.act = act

    def __call__(self,x):
        self.inputs = x
        y = x @ self.W + self.b

        return self.act(y)
    

    def backward(self,dout):

        dW = self.inputs.T @ dout
        db = np.sum(dout)

        self.W -= self.lr * dW
        self.b -= self.lr * db

    def pred(self,x):
        y = self(x)
        return (y>0.5)*1
        
    def fit(self,x,y,epochs=20):

        for i in range(epochs):
            
            out = self(x)
            dout = out - y
            self.backward(dout)
            loss = -np.sum(y*np.log(out))
            if(i % 20==0):
                print(f'epoch = {i}   loss = {loss:.2f}')

np.set_printoptions(precision=2,suppress=True)
np.random.seed(123)


x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

ppt = Perceptron(act=sigmoid)
ppt.lr = 0.1
pred = ppt(x)

ppt.fit(x,y,epochs=200)
pred = ppt.pred(x)

print(f'\nlabels  {y}')
print(f'pred    {pred}')

