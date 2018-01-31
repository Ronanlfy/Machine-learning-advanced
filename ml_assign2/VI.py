import numpy as np
from scipy.stats import norm
from scipy.stats import gamma
import matplotlib.pyplot as plt
from numpy import random

#using variational distribution 
def vari_post(mu_n,labda_n,a_n,b_n,mu,tau):
	p1 = norm(mu_n, 1 / labda_n).pdf(mu)
	p2 = gamma.pdf(tau, a_n, loc = 0, scale = 1/b_n)

	return p1*p2


def exact_post(mu_s,labda_s,a_s,b_s,mu,tau):
	var = 1 / (labda_s*tau)
	p1 = norm(mu_s, var).pdf(mu)
	p2 = gamma.pdf(tau, a_s, loc = 0, scale = 1/b_s)
	return p1*p2


def calculate(labda_0,mu_0,a_0,b_0,x):
	n = len(x)
	sum_x = sum(x)
	mean_x = np.mean(x)
	sub = [i - mean_x for i in x]
	sub_square = [j**2 for j in sub]

#parameters for exact posterior
	a_s = (n + 1) / 2 + a_0
	b_s = 0.5*(sum(sub_square) + n*labda_0*(mean_x - mu_0)**2 / (n+labda_0)) + b_0
	labda_s = n + labda_0
	mu_s = (sum_x+labda_0*mu_0) / labda_s

	return a_s,b_s,labda_s,mu_s

def update(labda_0,mu_0,a_0,b_0,mu_ni,labda_ni,x):
	n = len(x)
	sum_x = sum(x)
	mean_x = np.mean(x)
	square = [i**2 for i in x]
	sum_square = sum(square)

	a_n = (n + 1) / 2 + a_0
	b_n = 0.5*(sum_square - 2*(sum_x+labda_0*mu_0)*mu_ni + (n+labda_0)*(mu_ni**2+1/labda_ni) + labda_0*mu_0**2) + b_0
	#b_n = a_n * n / sum(sub_square)
	mu_n = (sum_x + labda_0 * mu_0) / (n + labda_0)
	labda_n = (a_n/b_n) * (n + labda_0)

	return a_n,b_n,labda_n,mu_n

def dataGen(N):
	data = random.rand(N)
	data = [i - np.mean(data) for i in data]
	data = [i / np.var(data)**0.5 for i in data]

	return data


#main function:
#choose a really simple initial case
a_0 = 2
b_0 = 5
mu_0 = 0
labda_0 = 0
dim = 100
N = 20

#also simple case for update
mu_n = 0.2
labda_n = 0.2
a_n = 0.5
b_n = 0.5
iteration = 3

mu = np.linspace(-2.0, 2.0, dim)
tau = np.linspace(-2.0, 2.0, dim)
x = dataGen(N)

#calculate the exact posterior:
p_exact =  np.zeros((dim,dim), dtype=float)
a_s,b_s,labda_s,mu_s = calculate(labda_0,mu_0,a_0,b_0,x)
for i in range(dim):
	for j in range(dim):
		p_exact[i,j] = exact_post(mu_s,labda_s,a_s,b_s,mu[j],tau[i])

#update
for i in range(iteration):
	a_n,b_n,labda_n,mu_n = update(labda_0,mu_0,a_0,b_0,mu_n,labda_n,x)
#calculate the variational posterior:
p_vari =  np.zeros((dim,dim), dtype=float)
for i in range(dim):
	for j in range(dim):
		p_vari[i,j] = vari_post(mu_n,labda_n,a_n,b_n,mu[j],tau[i])

#plot
[mu_show, tau_show] = np.meshgrid(mu,tau)
plt.figure()
exact_posterior = plt.contour(mu_show,tau_show,p_exact,colors='g')
variational_posterior = plt.contour(mu_show,tau_show,p_vari,colors='r')
plt.clabel(exact_posterior, fontsize=9, inline=1)
plt.clabel(variational_posterior, fontsize=9, inline=1)
plt.xlabel('$\mu$')
plt.ylabel('$\tau$')
plt.title('a0:{} b0:{} mu0:{} lambda0:{}'.format(a_0, b_0, mu_0,labda_0))
plt.show()


