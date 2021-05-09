import numpy as np
import matplotlib.pyplot as plt
import itertools
from math import ceil

class v_Rep:
    def __init__(self,j,ind):
        self.j=j
        self.ind=ind
    
    def comp(self,x):
        if self.ind==0:
            return np.array([0,DPhi(x,self.j,3),-DPhi(x,self.j,2)])
        elif self.ind==1:
            return np.array([-DPhi(x,self.j,3),0,DPhi(x,self.j,1)])
        else:
            return np.array([DPhi(x,self.j,2),-DPhi(x,self.j,1),0])

def DPhi(x, j, d, D=3):
    ret=1
    for i in range(D):
        if i != d-1:
            ret = ret * 0.5*np.sin(x[i]*np.pi*j[i])
        else:
            ret = ret * 0.5*np.pi*j[i]*np.cos(x[i]*np.pi*j[i])
    return ret

def sortFunc(x):
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]

def generateJs(k_max=1000):
    j_len = ceil(k_max/3)
    js = list(itertools.combinations_with_replacement(range(1,ceil(np.sqrt(j_len))), 3))
    js.sort(key=sortFunc)
    js = js[:j_len]
    return js

js=generateJs(40)
vs=np.empty(3*len(js), dtype=object)
for i,j in enumerate(js):
    vs[3*i]=v_Rep(j,0)
    vs[3*i+1]=v_Rep(j,1)
    vs[3*i+2]=v_Rep(j,2)

n=20
for ind in range(0, len(vs), 3):
    print("k: {}, j: {}".format(ind, vs[ind].j))
    x,y = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))

    u = np.zeros((n,n))
    w = np.zeros((n,n))
    fig = plt.figure(figsize=(10,4))
    for k in range(3):
        if ind+k >= len(vs):
            continue
        plt.subplot(1, 3, k+1)
        for i in range(n):
            for j in range(n):
                u[i][j], w[i][j], _ = vs[ind+k].comp([x[i][j],y[i][j],0.5])
        M=np.sqrt(u*u+w*w)
        plt.quiver(x,y,u,w,M,cmap=plt.cm.jet)
    plt.show()

""" n=10
x,y,z = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n),np.linspace(0,1,n))

u = np.zeros((n,n,n))
v = np.zeros((n,n,n))
w = np.zeros((n,n,n))
for i in range(n):
    for j in range(n):
        for k in range(n):
            u[i][j][k], v[i][j][k], w[i][j][k] = vs[1].comp([x[i][j][k],y[i][j][k],z[i][j][k]])
M=np.sqrt(u*u+v*v+w*w)
fig = plt.figure()
ax = fig.gca(projection='3d')
qq=plt.quiver(x,y,z,u,v,w)
plt.show() """