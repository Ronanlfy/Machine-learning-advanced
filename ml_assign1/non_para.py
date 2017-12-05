import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from numpy import random

def SamplePrior(sigma,l,num):
    x = np.arange(-5, 5, 0.01)
    n = len(x)
    mu = np.zeros((n))
    kernal = ComputeDist(x,x,l,sigma) 
    samples = np.random.multivariate_normal(mu,kernal,num)
    for i in range(num):
        plt.plot(x,samples[i,:])
    plt.title('length-scale: '+str(l))

def ComputeDist(x,y,l,sigma,beta):
    X = np.transpose([x])
    Y = np.transpose([y])
 #   X = x[:,None]
  #  Y = y[:,None]
    dis = cdist(X,Y,'sqeuclidean')
    if beta == 'None':
        kernal = np.exp(-dis/(l*l)) * (sigma*sigma)
    else:
        kernal = np.exp(-dis/(l*l)) * (sigma*sigma) + beta * np.eye(len(X))
    return kernal 

def ComputePos(x,xlist,ylist,l,sigma,beta):
    Cn = ComputeDist(xlist, xlist, l, sigma, beta) 
    ylist = np.transpose([ylist])
    k = ComputeDist(x, xlist, l, sigma, 'None')
    mu = np.dot(np.dot(k,np.linalg.inv(Cn)),ylist)
    var = ComputeDist(x,x,l,sigma,beta) 
    var = var - np.dot(np.dot(k,np.linalg.inv(Cn)),np.transpose(k)) 
    return mu,var

def plotPos(X,Y,l,sigma,beta):
    num = 400
    x = np.linspace(-0.5*np.pi, 2.5*np.pi, num)
    mu, var = ComputePos(x,X,Y,l,sigma,beta)
    #print(mu.shape)
    plt.figure()
    #plot posterior
    #plt.subplot(1,2,1)
    #plt.plot(X, Y,'ko')
    #plt.plot(x,np.cos(x),'r')
    #plt.plot(x,mu+2*np.sqrt(var),label="$UpperBound$", color="b")
    #plt.plot(x,mu-2*np.sqrt(var),label="$LowerBound$", color="g")
    #plt.title('posterior: length-scale: '+str(l))
    #plt.legend()
    #plot samples
    #plt.subplot(1,2,2)
    plt.plot(X, Y,'ko')
    plt.plot(x,np.cos(x),'r')
    mu = np.reshape(mu,(num,))
    Sam = np.random.multivariate_normal(mu,var,30)
    for i in range(30):
        plt.plot(x[:],Sam[i,:],linewidth=1)
    plt.title('posterior samples: length-scale: '+str(l))
    plt.show()

def plotPosSim(X,Y,l,sigma,beta):
    num = 400
    x = np.linspace(-0.5*np.pi, 2.5*np.pi, num)
    mu, var = ComputePos(x,X,Y,l,sigma,beta)
    plt.figure()
    #plot posterior
    plt.plot(X, Y,'ko')
    plt.plot(x,np.cos(x),label="$TrueCos$",color='r')
    #plt.plot(x,mu+2*np.sqrt(var),label="$UpperBound$", color="b")
    plt.plot(x,mu,label="$mean$", color="b")
    #plt.plot(x,mu-2*np.sqrt(var),label="$LowerBound$", color="g")
    plt.title('posterior : length-scale: '+str(l))
    plt.legend()
    plt.show()

#prior
sigma = 1
#num = 15
#l = [0.1,0.5,3,15]
#plt.figure()
#for i in range(len(l)):
 #   plt.subplot(2,2,i+1)
 #   SamplePrior(sigma,l[i],num)
#plt.show()

#formulate the points
l = [2.8,3,3.2]
n = 7
beta = 0.5
xlist = np.linspace(0, 2*np.pi, n)
ylist = np.cos(xlist) + random.normal(0, beta, n)
for i in range(len(l)):
    plotPos(xlist,ylist,l[i],sigma,0)
    plotPosSim(xlist,ylist,l[i],sigma,beta)
    



     




     
