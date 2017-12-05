import pylab as pb
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from numpy import random
from math import exp, sqrt, pi
from create_index_set import create_index_set

def bin2int(a):
	get_bin = lambda x, n: format(x, 'b').zfill(n)
	ans = get_bin(a,9)
	sep = [ans[i:(i+1)] for i in range(len(ans))]
	data = np.array(list(map(int, sep)))
	data = 2*data - 1
	#print(data)
	return data

def generateData():
	data = np.arange(0, 511, 1)
	dataSet = np.zeros((512,9))
	for i in range(len(data)):
		Set = bin2int(data[i])
		dataSet[i,:] = Set[::-1]
	return dataSet

def Visua(input):
	input = np.reshape(input,(3,3))
	tmp_str = ''
	for i in range(input.shape[0]):
		for j in range(input.shape[1]):
			tmp_str += ('x' if (input[i,j]==1) else 'o')
		tmp_str += '\n'

	print(tmp_str) 

def calpro(t, theta, model):
	t = np.reshape(t,(3,3))
	p = 1
	if model == 0:
		p = 1/512
	else:
		for i in range(3):
			for j in range(3):
				if model == 1:
					e = t[i,j]*theta[0]*(j-1)
					p *= 1/(1+exp(-e))
				elif model == 2:
					e = t[i,j]*(theta[0]*(j-1)+theta[1]*(1-i))
					p *= 1/(1+exp(-e))
				elif model == 3:
					e = t[i,j]*(theta[0]*(j-1)+theta[1]*(1-i)+theta[2])
					p *= 1/(1+exp(-e))
				else:
					print('model choice is out of range!')

	return p

def prior(sigma,model,N):
	mu = np.zeros(model)
	var = sigma*np.eye(model)
	#mu = 5*np.ones(model)
	#var = random.rand(model*model)
	#var = sigma*np.reshape(var,(model,model))
	samples = random.multivariate_normal(mu,var,N)
	return samples

def calevid(t ,sigma, model, N):
	samples = prior(sigma,model,N)
	evidence = 0
	for i in range(N):
		evidence += calpro(t, samples[i,:], model)
	return evidence/N

#main
dataSet = generateData()
#example = dataSet[2,:]
#print(example)
#Visua(example)
sigma = 1000
N = 10000
evi = np.zeros((4,512))
evi[0,:] = np.ones((512))/512
for i in range(512):
	evi[1,i] = calevid(dataSet[i,:] ,sigma, 1, N)
	evi[2,i] = calevid(dataSet[i,:] ,sigma, 2, N)
	evi[3,i] = calevid(dataSet[i,:] ,sigma, 3, N)
evi = np.transpose(evi)
#print(evi)
#print(evi.shape)

#max = np.argmax(evi,axis=0)
#min = np.argmin(evi,axis=0)
#print(max)
#for i in range(3):
	#print('most probability mass of model ' +str(i+1))
	#Visua(dataSet[max[i+1]])
	#print('least probability mass of model ' +str(i+1))
	#Visua(dataSet[min[i+1]])

#sum = np.sum(evi,axis=0)
#print('sum of the evidence for the whole dataset:')
#print(sum)

index = create_index_set(evi)
#print(index)
#print(index.shape)
plt.figure()
plt.plot(evi[index,3],'g', label= "P(D|M3)")
plt.plot(evi[index,2],'r', label= "P(D|M2)")
plt.plot(evi[index,1],'b', label= "P(D|M1)")
plt.plot(evi[index,0],'m--', label = "P(D|M0)")
plt.xlim(0,520)
plt.ylim(0,0.12)
plt.xlabel('All dataSet')
plt.ylabel('evidence')
plt.title('evidence of all data sets')
plt.legend()
plt.show()

plt.figure()
plt.plot(evi[index,3],'g', label= "P(D|M3)")
plt.plot(evi[index,2],'r', label= "P(D|M2)")
plt.plot(evi[index,1],'b', label= "P(D|M1)")
plt.plot(evi[index,0],'m--', label = "P(D|M0)")
plt.xlim(0,100)
plt.ylabel('subset of possible dataSet')
plt.ylabel('evidence')
plt.ylim(0,0.12)
plt.title('evidence of subset of possible data sets')
plt.legend()
plt.show()