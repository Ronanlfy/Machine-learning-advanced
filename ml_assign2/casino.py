import numpy as np
import matplotlib.pyplot as plt
  
def rolldice(index):
	dice1 = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
	dice2 = [1/3, 1/3, 1/3, 0, 0, 0]
	dice3 = [1/4, 1/4, 1/4, 1/4, 0, 0]
	dice4 = [1/4, 1/4, 1/4, 1/12, 1/12, 1/12]
	dice5 = [1/2, 1/6, 1/6, 0, 1/12, 1/12]
	dice6 = [1/4, 1/12, 1/6, 1/4, 1/12, 1/6]
	dice = np.array([dice1,dice2,dice3,dice4,dice5,dice6])
	dicepro = dice[index,:]
	#print(dicepro)
	return np.random.choice(6, 1, p=dicepro)

def TableSeq(K):
	table = np.zeros(K)
	table[0] = (np.random.rand() > 0.5) + 1
	for i in range(K-1):
		if table[i] == 1:
			table[i+1] = np.random.choice(2, 1, p=[0.25, 0.75]) + 1
		else:
			table[i+1] = np.random.choice(2, 1, p=[0.75, 0.25]) + 1

	return table

def DiceAssign(n):
# n means how many different dices we have
	pro = np.ones(n) / n
	dice1 = np.random.choice(n, 1, p=pro)
	dice2 = np.random.choice(n, 1, p=pro)
	#print(dice1)
	#print(dice2)

	return dice1[0],dice2[0]

def Casino(K,n):
	num = 1
	p = 0.5
	X = np.zeros((n,K))
	X_out = np.zeros((n,K))
	#dice1, dice2 = DiceAssign(num)
	dice1 = 1
	dice2 = 2
	obs = []
	#print(table1)
	for i in range(n):
		Seq = TableSeq(K)
		#print(Seq)
		for j in range(K):
			if Seq[j] == 1:
				X[i,j] = rolldice(dice1) + 1
			else:
				X[i,j] = rolldice(dice2) + 1
			hid = np.random.choice(2, 1, p=[p, 1-p])
			if hid == 0:
				X_out[i,j] = X[i,j]
			else:
				X_out[i,j] = 0

		S_temp = np.cumsum(X[i,:])
		S_temp = S_temp[K-1]
		obs.append(S_temp)
		X_output = X_out[i,:]
		X_output = [str(int(i)) for i in X_output]
		X_output = [num.replace('0','?') for num in X_output]
		obs.append(X_output)

	S = np.cumsum(X, axis = 1)
	S = S[:,K-1]

	return X, X_out, S


#main:
#K is the number of tables, n is the number of players
K = 20
n = 10000
X, X_out, S = Casino(K,n)
print(X)
#verify correctness
plt.figure()
plt.hist(S, bins = 200, color = "black", normed = False,) 
plt.xlabel('Output Sum')
plt.ylabel('Frequency')
plt.title('group1 use dice2, group2 use dice3')
plt.show()
