import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from numpy import random

def calprob(point, mu, sigma):
    pro = multivariate_normal(mu,sigma).pdf(point)
    return pro

def drawpro(data):
    plt.figure()
    plt.imshow(data,cmap='jet',extent=(-2, 2, -2, 2))
    #plt.colorbar()
    plt.xlabel('w0') 
    plt.ylabel('w1')
    plt.show()

def calprior(mu, sigma):
    w0 = np.arange(-2.0, 2.0, 0.02)
    w1 = np.arange(-2.0, 2.0, 0.02)
    x, y = np.meshgrid(w0, w1)
    pro = np.zeros((len(w0),len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            pro[i,j] = multivariate_normal(mu, sigma).pdf([x[i,j],y[i,j]])
    return pro

def drawline(w0,w1):
    x = np.arange(-2.0, 2.0, 0.02)
    fig = plt.figure()
    ax = plt.gca()
    for i in range(len(w0)):
        y = w0[i]*x + w1[i]
        plt.plot(x,y)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()

#classical resampling method for particle filter
def resampling(pro,num):
    length = len(pro)
    weights = np.reshape(pro,length*length)
    weights = weights/sum(weights)
    i, j = 0, 0
    N = len(weights) - 1
    # make N subdivisions, and choose positions with a consistent random offset
    positions = (random.random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    w0 = []
    w1 = []
    for a in range(num):
        ran = random.randint(0, N-1)
        index = indexes[ran] + 1
        x = int(index/length) 
        w1.append(x * 0.02 - 2)
        y = index - int(index/length)*length 
        w0.append(y * 0.02 - 2)
    return w0,w1
    
def calposterior(mu,sigma,sig_sqr,xlist,tlist,Z):
    w0 = np.arange(-2.0, 2.0, 0.02)
    w1 = np.arange(-2.0, 2.0, 0.02)
    n = len(w0) 
    x, y = np.meshgrid(w0, w1)
    index = random.randint(0, n-1)
    pro = np.zeros((n,n))
    for i in range(n):
        for j in range(len(w1)):
            pro[i,j] = multivariate_normal(mu, sigma).pdf([x[i,j],y[i,j]]) * norm(xlist[index]*x[i,j]+y[i,j], sig_sqr).pdf(tlist[index])
    if Z == 1:
        pro = pro
    else:
        Z = Z - 1
        for z in range(Z):
            index = random.randint(0, n-1)
            for i in range(len(w0)):
                for j in range(len(w1)):
                    pro[i,j] = pro[i,j] * norm(xlist[index]*x[i,j]+y[i,j], sig_sqr).pdf(tlist[index])
    return pro
   # return pro, samples      
        
 #   x = np.array([[x,1]])
 #   I = np.array([[1.0,0.0],[0.0,1.0]])
 #   sigma = np.dot(x.T,x)/sig_sqr + I/tau_sqr
 #   mu = np.dot(sigma, np.transpose(x)*t/sig_sqr)
 #   sigma = np.linalg.inv(sigma)
 #   mu = np.transpose(mu)
 #   pro = calprob(mu[0],sigma)
 #   print(mu[0])
 #   print(sigma)
 

#fomulate the datasets
n = 200
num = 15
xList = np.arange(-2, 2, 0.02)
tList = []
w = np.array([[1.5, -0.8]])
noise = random.normal(0,0.2,n)
for i in range(n):
    t = w[0,0]*xList[i] + w[0,1] + noise[i]
    tList.append(t)

#prior
mu = [0,0]
sig_sqr = 0.2
sigma = [[0.5,0],[0,0.5]]
#visialize the prior distribution
pro_prior = calprior(mu, sigma)
#drawpro(pro_prior)

#update 10 times
pro_new = calposterior(mu,sigma,sig_sqr,xList,tList,20)
w0,w1 = resampling(pro_new,num)
drawpro(pro_new)
drawline(w0,w1)









