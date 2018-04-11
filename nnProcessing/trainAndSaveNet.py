import numpy as np
import os
import pickle

VALIDATION_SPLIT = 0.1
DATASET_FNAME = './dataset/faceDataset.pickle'
data = []
#import the dataset
with open(DATASET_FNAME, 'rb') as f:
	data = pickle.load(f) 
X = data[:,:-6]
y = data[:,-6:]
print X.shape

def activate(Z, activation = 'relu'):
	if activation == 'sigmoid':
		expo = np.exp(-1 * Z)
		return 1.0/(1 + expo)
	elif activation == 'relu':
		Z[Z < 0] = 0
		return Z
	elif activation == 'softmax':
		A = np.zeros_like(Z)
		for sample in range(Z.shape[1]):
			print Z[:,sample]
			#exponentiate all 6 z's  and divide by sum of the exp(z's):
			expSample = np.exp(Z[:,sample])
			A[:,sample] = expSample/np.sum(expSample) 

def update(parameter, derivative, learning_rate, update_rule = 'grad_descent'):
	if update_rule == 'grad_descent':
		return parameter - LR * derivative

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

numVal = int(VALIDATION_SPLIT * X.shape[0])
X_train = X[:-numVal]
y_train = y[:-numVal]
X_test = X[-numVal:]
y_test = y[-numVal:]

print X_train.shape

#the network:
nAttributes = X_train.shape[1]
nSamples = X_train.shape[0]
inputs = X_train.T

nodesHL1 = 20
nodesHL2 = 10
nodesOUT = 6 #softmax here to get 6-long vector of confidences!

activationHL1 = 'relu'
activationHL2 = 'sigmoid'
activationOUT = 'softmax'

update_rule = 'grad_descent'

W_HL1 = np.random.rand(nodesHL1, nAttributes) #20x136
b_HL1 = np.zeros((nodesHL1, 1)) #20x1
W_HL2 = np.random.rand(nodesHL2, nodesHL1) #10x20
b_HL2 = np.zeros((nodesHL2, 1)) #10x1
W_OUT = np.random.rand(nodesOUT, nodesHL2) #6x10
b_OUT = np.zeros((nodesOUT, 1)) #6x1

NUM_EPOCHS = 1500
LR = 0.003
X_train = X_train.T #from 191x136 to 136x191, where each sample is a column
y_train = y_train.T #from 191x6 to 6x191, where each sample's label is a column
z_HL1 = []
a_HL1 = []
z_HL2 = []
a_HL2 = []
z_OUT = []
a_OUT = []
for i in range(NUM_EPOCHS):

	#forward propagation::
	# print b_HL1.shape
	z_HL1 = np.dot(W_HL1, X_train) + b_HL1 #20x136 * 136xb + 20x1 = 20xb
	a_HL1 = activate(z_HL1, activationHL1) #20xb
	
	z_HL2 = np.dot(W_HL2, a_HL1) + b_HL2 #10x20 * 20xb + 10xb = 10xb
	a_HL2 = activate(z_HL2, activationHL2) #10xb
	
	z_OUT = np.dot(W_OUT, a_HL2) + b_OUT #6x10 * 10xb + 6x1 = 6xb
	a_OUT = activate(np.array(z_OUT,dtype = np.float32), activationOUT) #6xb

	#loss computation::
	# loss = np.sum(-y_train.T * np.log(a_OUT) -(1 - y_train.T) * np.log(1 - a_OUT)) / nSamples
	loss = np.sum(np.sum(-y_train * np.log(a_OUT), axis = 0)) #sum over the cross entropy of every sample for every label, then sum over all samples!
	#replaced with cross entropy loss!
	print 'Epoch : {}   Loss : {}'.format(i,loss)
	exit()
	#backward propagation::
	
	da_OUT = -np.sum(y_train - a_OUT) / nSamples#single valued
	dz_OUT = a_OUT * (1 - a_OUT) * da_OUT #1xb
	dW_OUT = np.dot(dz_OUT, a_HL2.T) #1xb * bx3 = 1x3
	db_OUT = np.sum(dz_OUT)
	
	da_HL2 = np.dot(W_OUT.T, dz_OUT) #3x1 * 1xb = 3xb
	drelu = np.zeros_like(da_HL2)
	drelu[a_HL2 > 0] = 1
	dz_HL2 = drelu * da_HL2 # 3xb * 3xb one to one
	dW_HL2 = np.dot(dz_HL2, a_HL1.T) #3xb * bx3 = 3x3
	db_HL2 = np.reshape(np.sum(dz_HL2, axis = 1),(3,1)) #3x1
	
	da_HL1 = np.dot(W_HL2.T, dz_HL2) # 3x3 * 3xb = 3xb
	drelu = np.zeros_like(da_HL1) #3xb
	drelu[a_HL1 > 0] = 1
	dz_HL1 = drelu * da_HL1 #3xb * 3xb one to one
	dW_HL1 = np.dot(dz_HL1, X_train.T)  # 3xb * bx7 = 3x7
	db_HL1 = np.reshape(np.sum(dz_HL1, axis = 1),(3,1))
	
	#weight update::
	W_HL1 = update(W_HL1, dW_HL1, LR, update_rule)
	b_HL1 = update(b_HL1, db_HL1, LR, update_rule)
	
	W_HL2 = update(W_HL2, dW_HL2, LR, update_rule)
	b_HL2 = update(b_HL2, db_HL2, LR, update_rule)
	
	W_OUT = update(W_OUT, dW_OUT, LR, update_rule)
	b_OUT = update(b_OUT, db_OUT, LR, update_rule)

#evaluate validation accuracy:
z_HL1 = np.dot(W_HL1, X_test.T) + b_HL1 #3x7 * 7xb + 3x1 = 3xb
a_HL1 = activate(z_HL1, activationHL1) #3xb
z_HL2 = np.dot(W_HL2, a_HL1) + b_HL2 #3x3 * 3xb + 3xb = 3xb
a_HL2 = activate(z_HL2, activationHL2) #3xb
z_OUT = np.dot(W_OUT, a_HL2) + b_OUT #1x3 * 3xb + 1x1 = 1xb
a_OUT = activate(np.array(z_OUT,dtype = np.float32), activationOUT) #1xb

correct = 0
for i in range(numVal):
	if a_OUT[0][i] >= 0.5 and y_test[i] == 1:
		correct += 1.0
	if a_OUT[0][i] < 0.5 and y_test[i] == 0:
		correct += 1.0	
print correct / numVal