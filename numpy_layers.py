"""Explicit batch/layer normalization and normalization-aware dense backpropagation."""
import numpy as np

def matrix(x,width):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or not len(x) or x.shape[1]!=width or not np.isfinite(x).all():raise ValueError('Expected finite nonempty matrix of fitted width')
    return x

def targets(y,count):
    y=np.asarray(y)
    if y.shape!=(count,) or not np.isin(y,[0,1]).all():raise ValueError('Binary target shape/value mismatch')
    return y

class BatchNorm1d:
    def __init__(self,n_features,eps=1e-5,momentum=.1):
        if type(n_features) is not int or n_features<1 or not np.isfinite([eps,momentum]).all() or eps<=0 or not 0<=momentum<=1:raise ValueError('Invalid normalization settings')
        self.n_features,self.eps,self.momentum=n_features,eps,momentum
        self.gamma=np.ones(n_features);self.beta=np.zeros(n_features)
        self.running_mean=np.zeros(n_features);self.running_var=np.ones(n_features);self._cache=None
    def forward(self,x,training=True):
        x=matrix(x,self.n_features)
        if training:
            mean=x.mean(axis=0);variance=x.var(axis=0)
            if not np.isfinite(mean).all() or not np.isfinite(variance).all():raise ValueError('Normalization overflow')
            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*mean
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*variance
        else:mean,variance=self.running_mean,self.running_var
        normalized=(x-mean)/np.sqrt(variance+self.eps)
        if not np.isfinite(normalized).all():raise ValueError('Invalid normalization statistics')
        self._cache=(normalized,variance) if training else None
        return self.gamma*normalized+self.beta
    def backward(self,gradient):
        if self._cache is None:raise ValueError('A training forward pass is required')
        gradient=matrix(gradient,self.n_features);normalized,variance=self._cache
        if gradient.shape!=normalized.shape:raise ValueError('Gradient shape mismatch')
        dgamma=(gradient*normalized).sum(axis=0);dbeta=gradient.sum(axis=0)
        dxhat=gradient*self.gamma;n=len(gradient)
        dx=(n*dxhat-dxhat.sum(axis=0)-normalized*(dxhat*normalized).sum(axis=0))/n/np.sqrt(variance+self.eps)
        return dx,dgamma,dbeta

class LayerNorm1d:
    def __init__(self,n_features,eps=1e-5):
        if type(n_features) is not int or n_features<1 or not np.isfinite(eps) or eps<=0:raise ValueError('Invalid normalization settings')
        self.n_features,self.eps=n_features,eps;self.gamma=np.ones(n_features);self.beta=np.zeros(n_features);self._cache=None
    def forward(self,x):
        x=matrix(x,self.n_features);mean=x.mean(axis=1,keepdims=True);variance=x.var(axis=1,keepdims=True)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():raise ValueError('Normalization overflow')
        normalized=(x-mean)/np.sqrt(variance+self.eps);self._cache=(normalized,variance)
        return self.gamma*normalized+self.beta
    def backward(self,gradient):
        if self._cache is None:raise ValueError('A forward pass is required')
        gradient=matrix(gradient,self.n_features);normalized,variance=self._cache
        if gradient.shape!=normalized.shape:raise ValueError('Gradient shape mismatch')
        dxhat=gradient*self.gamma;n=self.n_features
        dx=(n*dxhat-dxhat.sum(axis=1,keepdims=True)-normalized*(dxhat*normalized).sum(axis=1,keepdims=True))/n/np.sqrt(variance+self.eps)
        return dx,(gradient*normalized).sum(axis=0),gradient.sum(axis=0)

class MLPWithBN:
    def __init__(self,use_bn=True,lr=.05,seed=42,input_dim=30,hidden_dims=(64,32,16)):
        dims=[input_dim,*hidden_dims,1]
        if type(use_bn) is not bool or any(type(d) is not int or d<1 for d in dims) or not np.isfinite(lr) or lr<=0:raise ValueError('Invalid network dimensions or learning rate')
        self.use_bn,self.lr=use_bn,lr;self.dims=dims;rng=np.random.default_rng(seed)
        self.W=[rng.normal(size=(b,a))*np.sqrt(2/a) for a,b in zip(dims[:-1],dims[1:])]
        self.b=[np.zeros((width,1)) for width in dims[1:]]
        self.bn_layers=[BatchNorm1d(width) for width in dims[1:-1]] if use_bn else []
        self._training=False
    def forward(self,x,training=True):
        x=matrix(x,self.dims[0]);self.cache_Z=[];self.cache_A=[x.T];self._training=training
        for i in range(len(self.W)):
            z=self.W[i]@self.cache_A[-1]+self.b[i]
            if not np.isfinite(z).all():raise ValueError('Activation overflow')
            self.cache_Z.append(z)
            if i==len(self.W)-1:a=np.exp(-np.logaddexp(0,-z))
            else:
                a=np.maximum(0,z)
                if self.use_bn:a=self.bn_layers[i].forward(a.T,training).T
            self.cache_A.append(a)
        return self.cache_A[-1].T
    def gradients(self,y):
        if not self._training:raise ValueError('A training forward pass is required')
        y=targets(y,self.cache_A[0].shape[1])
        dz=(self.cache_A[-1]-y[None,:])/len(y)
        dw=[None]*len(self.W);db=[None]*len(self.W);dg=[None]*len(self.bn_layers);dt=[None]*len(self.bn_layers)
        for i in range(len(self.W)-1,-1,-1):
            if i<len(self.W)-1:
                if self.use_bn:
                    da,dg[i],dt[i]=self.bn_layers[i].backward(da.T);da=da.T
                dz=da*(self.cache_Z[i]>0)
            dw[i]=dz@self.cache_A[i].T;db[i]=dz.sum(axis=1,keepdims=True);da=self.W[i].T@dz
        return {'W':dw,'b':db,'gamma':dg,'beta':dt}
    def step(self,x,y):
        x=matrix(x,self.dims[0]);y=targets(y,len(x))
        self.forward(x,training=True);gradient=self.gradients(y)
        if any(not np.isfinite(g).all() for values in gradient.values() for g in values):raise ValueError('Gradient overflow')
        for name,parameters in [('W',self.W),('b',self.b),('gamma',[layer.gamma for layer in self.bn_layers]),('beta',[layer.beta for layer in self.bn_layers])]:
            for parameter,grad in zip(parameters,gradient[name]):parameter-=self.lr*np.clip(grad,-5,5)
    def compute_loss(self,x,y,training=False):
        x=matrix(x,self.dims[0]);y=targets(y,len(x));self.forward(x,training)
        logits=self.cache_Z[-1].ravel();return float(np.mean(np.logaddexp(0,logits)-y*logits))
    def accuracy(self,x,y):
        x=matrix(x,self.dims[0]);y=targets(y,len(x))
        return float(np.mean((self.forward(x,False).ravel()>=.5)==y))

class AdamOptimizer:
    def __init__(self,lr=.05,beta1=.9,beta2=.999,eps=1e-8):
        if not np.isfinite([lr,beta1,beta2,eps]).all() or lr<=0 or eps<=0 or not 0<=beta1<1 or not 0<=beta2<1:raise ValueError('Invalid optimizer settings')
        self.lr,self.beta1,self.beta2,self.eps=lr,beta1,beta2,eps;self.m=0.;self.v=0.;self.t=0
    def step(self,w,grad):
        if not np.isfinite([w,grad]).all():raise ValueError('Finite scalar parameter/gradient required')
        self.t+=1;self.m=self.beta1*self.m+(1-self.beta1)*grad;self.v=self.beta2*self.v+(1-self.beta2)*grad**2
        return w-self.lr*(self.m/(1-self.beta1**self.t))/(np.sqrt(self.v/(1-self.beta2**self.t))+self.eps)

class MomentumOptimizer:
    """Exponential-gradient momentum; velocity includes the retained (1-beta) factor."""
    def __init__(self,lr=.05,beta=.9):
        if not np.isfinite([lr,beta]).all() or lr<=0 or not 0<=beta<1:raise ValueError('Invalid momentum settings')
        self.lr,self.beta=lr,beta;self.v=0.
    def step(self,w,grad):
        if not np.isfinite([w,grad]).all():raise ValueError('Finite scalar parameter/gradient required')
        self.v=self.beta*self.v+(1-self.beta)*grad
        return w-self.lr*self.v

class RMSPropOptimizer:
    """Retained RMSProp variant uses epsilon inside the square root."""
    def __init__(self,lr=.01,rho=.9,eps=1e-8):
        if not np.isfinite([lr,rho,eps]).all() or lr<=0 or eps<=0 or not 0<=rho<1:raise ValueError('Invalid RMSProp settings')
        self.lr,self.rho,self.eps=lr,rho,eps;self.v=0.
    def step(self,w,grad):
        if not np.isfinite([w,grad]).all():raise ValueError('Finite scalar parameter/gradient required')
        self.v=self.rho*self.v+(1-self.rho)*grad**2
        return w-self.lr*grad/np.sqrt(self.v+self.eps)

def loss_fn(w):return (w-3)**2+.5*np.sin(5*w)
def grad_fn(w):return 2*(w-3)+2.5*np.cos(5*w)
