import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from numpy import random
import scipy.optimize as opt
from math import pi

def generateData():
	x = np.linspace(0, 4.0*np.pi, 100)
	t = np.zeros((100,2))
	t[:,0] = x*np.sin(x)
	t[:,1] = x*np.cos(x)
	#print(t)
	plt.figure()
	plt.plot(t[:,0],t[:,1], 'r.')
	plt.title("the actual X in two-dimensional space")
	plt.legend(["original X"])
	A = np.zeros((10,2))
	A[:,0] = random.normal(0,1,10)
	A[:,1] = random.normal(0,1,10)
	Y = np.dot(t,np.transpose(A))
	return x,Y

def f(W):
	D = 10
	L = 2
	N =100
	W = np.reshape(W,(D,L))
	C = sigma*np.eye(D) + np.dot(W,np.transpose(W))
	inv = np.linalg.inv(C)
	det = np.linalg.det(C)
	val = N*D*0.5*np.log(2*pi) + N*0.5*np.log(det) + 0.5*np.trace(np.dot(np.dot(Y,inv),np.transpose(Y)))
	return val

def dfx(W):
	D = 10
	L = 2
	N =100
	W = np.reshape(W,(D,L))
	C = sigma*np.eye(D) + np.dot(W,np.transpose(W))
	inv = np.linalg.inv(C)
	S = np.diag(np.diag(np.dot(np.transpose(Y),Y)))
	val = np.zeros((D,L))
	delta = np.zeros((D,D))

	for i in range(D):
		for j in range(L):
			m = np.zeros((D,L))
			m[i,j] = 1
			delta = np.dot(m,np.transpose(W)) + np.dot(W,np.transpose(m))
			val[i,j] = N*0.5*(np.trace(np.dot(inv,delta)) + np.trace(np.dot(np.dot(np.dot(-inv,delta),inv),S)))
			#delta = delta + np.dot(m,np.transpose(W)) + np.dot(W,np.transpose(m))

	#val = N*0.5*(np.trace(np.dot(inv,delta)) + np.trace(np.dot(np.dot(np.dot(-inv,delta),inv),S)))
	val = np.reshape(val,(D*L,))
	#print(val)
	return val

#main
D = 10
L = 2
N =100
x,Y_pure = generateData()
#print(Y_pure.shape)
sig = [0.1, 0.4, 1]
W0 = np.reshape(random.rand(20),(D,L))
print(W0)
for i in range(len(sig)):
	sigma = sig[i]
	Y = Y_pure
	#Y = Y_pure + random.multivariate_normal(np.zeros((D)), sigma*np.eye(D), N)
	Wstar =  opt.fmin_cg(f,W0,fprime=dfx)
	#print(Wstar) 
	Wstar = np.reshape(Wstar,(D,L))

	square = np.dot(np.transpose(Wstar),Wstar)
	t_recover = np.dot(np.dot(Y,Wstar),np.linalg.inv(square))
	#print(t_recover.shape)
	plt.figure()
	dot = plt.plot(t_recover[:,0],t_recover[:,1], 'r.')
	plt.title('the retrieve latent X with sigma: ' + str(sigma))
	plt.legend([dot],["Recovered X"])
	plt.show()