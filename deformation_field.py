import numpy as np
import matplotlib.pyplot as plt
import itertools
from math import ceil

k=1000

class DField:
    def __init__(self, cs, js, X):
        self.cs = cs
        self.js = js
        self.X = X

def B_phi(x, j):
    assert(x.shape[1] == j.shape[0])
    return 0.5 * np.prod(np.sin(np.pi*x*j[na,:]), axis=1)

def dB_phi_dxk(x,j,k):
    assert(x.shape[1] == 3 and j.shape[0] == 3 and (0<=k<3))
    l = (k+1)%3
    m = (k+2)%3
    return 0.5**3 * np.pi*j[k] * np.cos(np.pi*x[:,k]*j[k]) *  np.sin(np.pi*x[:,l]*j[l]) * np.sin(np.pi*x[:,m]*j[m])

def B_v(x,j):
    v1 = np.stack([ np.zeros(x.shape[0]), dB_phi_dxk(x,j,2),    -dB_phi_dxk(x,j,1)],    axis=-1)
    v2 = np.stack([-dB_phi_dxk(x,j,2),    np.zeros(x.shape[0]),  dB_phi_dxk(x,j,0)],    axis=-1)
    v3 = np.stack([ dB_phi_dxk(x,j,1),   -dB_phi_dxk(x,j,0),     np.zeros(x.shape[0])], axis=-1)
    return np.stack([v1,v2,v3], axis=-1)

def sortFunc(x):
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]

def generateJs(k_max=1000):
    j_len = ceil(k_max/3)
    js = list(itertools.product(range(1,ceil(np.sqrt(j_len))), repeat=3))
    js.sort(key=sortFunc)
    js = js[:j_len]
    return np.array(js)

js=generateJs(k)

def getDeformationField(coefficients, js, x):
    """
    Constructs a deformation field by adding the basis vectors weighted by the coefficients
    """
    sum=np.zeros(x.shape)
    for i in range(0, len(coefficients)-2, 3):
        bv = B_v(x, js[i//3])
        sum += coefficients[i]*bv[:,:,0]
        sum += coefficients[i+1]*bv[:,:,1]
        sum += coefficients[i+2]*bv[:,:,2]
    bv = B_v(x, js[(len(coefficients)-1)//3])
    if len(coefficients) % 3 == 1 or len(coefficients) % 3 == 2:
        sum += coefficients[len(coefficients)-2]*bv[:,:,0]
    if len(coefficients) % 3 == 2:
        sum += coefficients[len(coefficients)-1]*bv[:,:,1]
    return sum

def rungeKutta(dField, T=100):
    xn = dField.X
    h = 1/T
    for t in range(1,T+1):
        xn = xn + h*getDeformationField(dField.cs, dField.js, xn+(h/2)*getDeformationField(dField.cs, dField.js, xn))
    return xn

def EStep(fn, ym, sigmas):
    ds = calc_ds(fn, ym)
    numerator = np.exp(-1/(2*np.square(sigmas))*ds**2)
    denominator = (2*np.pi*np.square(sigmas))**1.5 + np.sum([np.exp(-1/(2*np.square(sigmas))*np.square(dn)) for dn in ds], axis=0)
    return numerator/denominator

def calc_ds(fn, ym):
    return np.linalg.norm(ym-fn, axis=1)**2

if __name__ == "__main__":

    """ n=10
    x,y,z = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n), np.linspace(0,1,n))
    X = np.stack([x,y,z],axis=-1).reshape(n*n*n, 3)
    cs = np.ones(33)
    f = getDeformationField(cs, X, js)
    u = f[:,0].reshape((n,n,n))
    v = f[:,1].reshape((n,n,n))
    w = f[:,2].reshape((n,n,n))
    uvnorms = np.linalg.norm(f, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.quiver(x,y,z,u,v,w,length=0.01,normalize=False)
    plt.show() """

    n=25
    x,y = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    X = np.stack([x,y,0.5*np.ones(x.shape)],axis=-1).reshape(n*n, 3)
    print(X.shape)
    cs = np.array([0,0,1])
    f = getDeformationField(cs, js, X)
    u = f[:,0].reshape((n,n))
    v = f[:,1].reshape((n,n))
    uvnorms = np.linalg.norm(f, axis=1)
    plt.quiver(x,y,u,v,uvnorms,cmap=plt.cm.jet)   
    plt.show()