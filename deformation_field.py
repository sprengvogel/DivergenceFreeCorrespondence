import numpy as np
import matplotlib.pyplot as plt
import itertools
from math import ceil
from tqdm import tqdm

HUBER_RADIUS = 1
SIGMA_SQUARE = 0.01

class DField:
    def __init__(self, cs, js):
        self.cs = cs
        self.js = js

"""def B_phi(x, j):
    assert(x.shape[1] == j.shape[0])
    return 0.5 * np.prod(np.sin(np.pi*x*j[na,:]), axis=1)"""

def dB_phi_dxk(x,j,k):
    assert(x.shape[1] == 3 and j.shape[0] == 3 and (0<=k<3))
    l = (k+1)%3
    m = (k+2)%3
    return 0.5**3 * np.pi*j[k] * np.cos(np.pi*x[:,k]*j[k]) *  np.sin(np.pi*x[:,l]*j[l]) * np.sin(np.pi*x[:,m]*j[m])

"""
def B_v(x,j):
    v1 = np.stack([ np.zeros(x.shape[0]), dB_phi_dxk(x,j,2),    -dB_phi_dxk(x,j,1)],    axis=-1)
    v2 = np.stack([-dB_phi_dxk(x,j,2),    np.zeros(x.shape[0]),  dB_phi_dxk(x,j,0)],    axis=-1)
    v3 = np.stack([ dB_phi_dxk(x,j,1),   -dB_phi_dxk(x,j,0),     np.zeros(x.shape[0])], axis=-1)
    return np.stack([v1,v2,v3], axis=-1)"""

def B_v(x,k,js):
    j = js[k//3]
    if k%3 == 0:
        return np.stack([ np.zeros(x.shape[0]), dB_phi_dxk(x,j,2),    -dB_phi_dxk(x,j,1)],    axis=-1)
    elif k%3 == 1:
        return np.stack([-dB_phi_dxk(x,j,2),    np.zeros(x.shape[0]),  dB_phi_dxk(x,j,0)],    axis=-1)
    else:
        return np.stack([ dB_phi_dxk(x,j,1),   -dB_phi_dxk(x,j,0),     np.zeros(x.shape[0])], axis=-1)

def Dxvk(x,k,js):
    j = js[k//3]
    Dxvk = np.zeros((x.shape[0],3,3))
    if k%3 == 0:
        Dxvk[:,1,0] = j[0]*np.cos(x[:,0]*np.pi*j[0])*np.sin(x[:,1]*np.pi*j[1])*np.cos(x[:,2]*np.pi*j[2])
        Dxvk[:,1,1] = j[1]*np.sin(x[:,0]*np.pi*j[0])*np.cos(x[:,1]*np.pi*j[1])*np.cos(x[:,2]*np.pi*j[2])
        Dxvk[:,1,2] = -j[2]*np.sin(x[:,0]*np.pi*j[0])*np.sin(x[:,1]*np.pi*j[1])*np.sin(x[:,2]*np.pi*j[2])

        Dxvk[:,2,0] = -j[0]*np.cos(x[:,0]*np.pi*j[0])*np.cos(x[:,1]*np.pi*j[1])*np.sin(x[:,2]*np.pi*j[2])
        Dxvk[:,2,1] = j[1]*np.sin(x[:,0]*np.pi*j[0])*np.sin(x[:,1]*np.pi*j[1])*np.sin(x[:,2]*np.pi*j[2])
        Dxvk[:,2,2] = -j[2]*np.sin(x[:,0]*np.pi*j[0])*np.cos(x[:,1]*np.pi*j[1])*np.cos(x[:,2]*np.pi*j[2])        
    elif k%3 == 1:
        Dxvk[:,0,0] = -j[0]*np.cos(x[:,0]*np.pi*j[0])*np.sin(x[:,1]*np.pi*j[1])*np.cos(x[:,2]*np.pi*j[2])
        Dxvk[:,0,1] = -j[1]*np.sin(x[:,0]*np.pi*j[0])*np.cos(x[:,1]*np.pi*j[1])*np.cos(x[:,2]*np.pi*j[2])
        Dxvk[:,0,2] = j[2]*np.sin(x[:,0]*np.pi*j[0])*np.sin(x[:,1]*np.pi*j[1])*np.sin(x[:,2]*np.pi*j[2])

        Dxvk[:,2,0] = -j[0]*np.sin(x[:,0]*np.pi*j[0])*np.sin(x[:,1]*np.pi*j[1])*np.sin(x[:,2]*np.pi*j[2])
        Dxvk[:,2,1] = j[1]*np.cos(x[:,0]*np.pi*j[0])*np.cos(x[:,1]*np.pi*j[1])*np.sin(x[:,2]*np.pi*j[2])
        Dxvk[:,2,2] = j[2]*np.cos(x[:,0]*np.pi*j[0])*np.sin(x[:,1]*np.pi*j[1])*np.cos(x[:,2]*np.pi*j[2])
    else:
        Dxvk[:,0,0] = j[0]*np.cos(x[:,0]*np.pi*j[0])*np.cos(x[:,1]*np.pi*j[1])*np.sin(x[:,2]*np.pi*j[2])
        Dxvk[:,0,1] = -j[1]*np.sin(x[:,0]*np.pi*j[0])*np.sin(x[:,1]*np.pi*j[1])*np.sin(x[:,2]*np.pi*j[2])
        Dxvk[:,0,2] = j[2]*np.sin(x[:,0]*np.pi*j[0])*np.cos(x[:,1]*np.pi*j[1])*np.cos(x[:,2]*np.pi*j[2])

        Dxvk[:,1,0] = j[0]*np.sin(x[:,0]*np.pi*j[0])*np.sin(x[:,1]*np.pi*j[1])*np.sin(x[:,2]*np.pi*j[2])
        Dxvk[:,1,1] = -j[1]*np.cos(x[:,0]*np.pi*j[0])*np.cos(x[:,1]*np.pi*j[1])*np.sin(x[:,2]*np.pi*j[2])
        Dxvk[:,1,2] = -j[2]*np.cos(x[:,0]*np.pi*j[0])*np.sin(x[:,1]*np.pi*j[1])*np.cos(x[:,2]*np.pi*j[2])
    Dxvk = 0.125*np.pi*np.pi*Dxvk
    return Dxvk

def sortFunc(x):
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]

def generateJs(k_max=1000):
    j_len = ceil(k_max/3)
    js = list(itertools.product(range(1,ceil(np.sqrt(j_len))), repeat=3))
    js.sort(key=sortFunc)
    js = js[:j_len]
    return np.array(js)

def getDeformationField(coefficients, js, x):
    """
    Constructs a deformation field by adding the basis vectors weighted by the coefficients
    """
    sum=np.zeros(x.shape)
    for k in range(len(coefficients)):
        bv = B_v(x, k, js)
        sum += coefficients[k]*bv
    return sum

def rungeKutta(dField, vertices, T=100):
    xn = vertices
    h = 1/T
    for t in range(1,T+1):
        xn = xn + h*getDeformationField(dField.cs, dField.js, xn+(h/2)*getDeformationField(dField.cs, dField.js, xn))
    return xn

"""
One Expectation Step iteration
"""
def eStep(fn, ym):
    ds = calc_ds(fn, ym)
    numerator = np.exp(-1/(2*SIGMA_SQUARE)*ds**2)
    denominator = (2*np.pi*SIGMA_SQUARE)**1.5 + np.sum([np.exp(-1/(2*SIGMA_SQUARE)*np.square(dn)) for dn in ds], axis=0)
    #Numerator is (n,m), denominator  (m,) and gets broadcasted to (n,m) in product
    res = numerator/denominator
    assert res.shape == (fn.shape[0],ym.shape[0])
    return res

def calc_ds(fn, ym):
    ds = np.zeros((fn.shape[0],ym.shape[0]))
    for i in range(fn.shape[0]):
        ds[i,:] = np.linalg.norm(ym-fn[i], axis=1)
    assert ds.shape == (fn.shape[0],ym.shape[0])
    return ds

"""
One Maximization Step iteration
"""
def mStep(dField, xn, ym, W):
    W = applyHuberLoss(W, xn, ym)
    fn, Dafn = calc_PartialDerivatives(dField, xn)
    J = np.reshape(Dafn, (xn.shape[0]*3, len(dField.cs)))
    r = calc_r(W, fn, ym)
    WSnake = calc_WSnake(W)
    LInv = calc_LInv(dField)
    dField.cs = dField.cs - np.linalg.inv(np.transpose(J)@WSnake@J+SIGMA_SQUARE*LInv)@(np.transpose(J)@r-SIGMA_SQUARE*(LInv@dField.cs))
    return dField

"""
Recursion to calculate f_n and D_a f_n
"""
def calc_PartialDerivatives(dField, vertices, T=100):
    xn = vertices
    #Daxn is (N,3,K) where K is number of coefficients
    Daxn = np.zeros((xn.shape[0],3,len(dField.cs)))
    h = 1/T
    for t in range(1,T+1):
        Daxn = daxn_step(Daxn, xn, dField, h)
        xn = xn + h*getDeformationField(dField.cs, dField.js, xn+(h/2)*getDeformationField(dField.cs, dField.js, xn))
    return xn, Daxn

"""
One step of the iteration to calculate D_a x_{n+1}
"""
def daxn_step(Daxn, xn, dField, h):
    dFieldResult = xn+(h/2)*getDeformationField(dField.cs, dField.js, xn)
    max_k = len(dField.cs)
    term2_sum = np.zeros((xn.shape[0],3,3))
    term3 = np.zeros((xn.shape[0],3,max_k))
    term4 = np.zeros((xn.shape[0],3,max_k))
    for k in range(max_k):
        term2_sum += Dxvk(xn,k,dField.js)*dField.cs[k]
        term3[:,:,k] = B_v(xn, k, dField.js)
        term4[:,:,k] = B_v(dFieldResult, k, dField.js)
    term2 = (np.identity(3)+(h/2)*term2_sum)@Daxn
    
    outer_sum = 0
    for k in range(max_k):
        term1 = Dxvk(dFieldResult, k, dField.js)
        outer_sum += term1@(term2+term3)*dField.cs[k]

    Daxn = Daxn + h*outer_sum + h*term4
    return Daxn

"""
Calculates diagonal matrix of eigenvalues L
"""
def calc_LInv(dField):
    L = np.zeros((dField.cs.shape[0],dField.cs.shape[0]))
    for k in range(len(dField.cs)):
        j = dField.js[k]
        L[k,k] = 1/np.power((np.pi**2)*np.sum(np.square(j)),-3/2)
    assert L.shape == (dField.cs.shape[0], dField.cs.shape[0])
    return L
    
"""
Calculates weighted residual vector r with shape(n*d,)
"""
def calc_r(W,fn,ym):
    n = fn.shape[0]
    r = np.zeros((3*n,))
    for i in range(n):
        r[3*i:3*i+3] = np.sum((W[i]*(fn[i]-ym).T),axis=1)
    assert r.shape == (3*n,)
    return r

"""
Calculates WSnake, diagonal matrix of column sums
"""
def calc_WSnake(W):
    sums = np.zeros(3*W.shape[1])
    for i in range(W.shape[1]):
        sums[3*i:3*i+3] = np.sum(W[i]) 
    WSnake = np.diag(sums)
    assert WSnake.shape == (3*W.shape[0],3*W.shape[0])
    return WSnake

def applyHuberLoss(W, fn, ym):
    for n in range(len(fn)):
        for m in range(len(ym)):
            if np.linalg.norm(fn[n]-ym[m]) > HUBER_RADIUS:
                W[n,m] *= HUBER_RADIUS/np.linalg.norm(fn[n]-ym[m])
    return W

if __name__ == "__main__":

    """n=10
    x,y,z = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n), np.linspace(0,1,n))
    X = np.stack([x,y,z],axis=-1).reshape(n*n*n, 3)
    cs = np.ones(3)
    f = getDeformationField(cs, X, js)
    u = f[:,0].reshape((n,n,n))
    v = f[:,1].reshape((n,n,n))
    w = f[:,2].reshape((n,n,n))
    uvnorms = np.linalg.norm(f, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.quiver(x,y,z,u,v,w,length=0.01,normalize=False)
    plt.show() """
    js=generateJs()

    n=25
    x,y = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    X = np.stack([x,y,0.5*np.ones(x.shape)],axis=-1).reshape(n*n, 3)
    print(X.shape)
    cs = np.ones(100)
    f = getDeformationField(cs, js, X)
    u = f[:,0].reshape((n,n))
    v = f[:,1].reshape((n,n))
    uvnorms = np.linalg.norm(f, axis=1)
    plt.quiver(x,y,u,v,uvnorms,cmap=plt.cm.viridis)   
    plt.show()