import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from scipy.stats import multivariate_normal


def drawpro(data):
    plt.figure()
    plt.imshow(data,cmap='jet',extent=(-100, 100, -100, 100))
    #plt.colorbar()
    plt.xlabel('w0') 
    plt.ylabel('w1')
    plt.title("normal distribution with variance=1000")
    plt.show()

def calprior(mu, sigma):
    w0 = np.arange(-100, 100, 2)
    w1 = np.arange(-100, 100, 2)
    x, y = np.meshgrid(w0, w1)
    pro = np.zeros((len(w0),len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            pro[i,j] = multivariate_normal(mu, sigma).pdf([x[i,j],y[i,j]])
    return pro

mu = [0,0]
sigma = 1000*np.eye(2)
pro_prior = calprior(mu, sigma)
drawpro(pro_prior)