from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import os
import pickle
import tensorflow as tf
VALIDATION_SPLIT = 0.1
DATASET_FNAME = './dataset/faceDataset.pickle'
data = []
#import the dataset
with open(DATASET_FNAME, 'rb') as f:
	data = pickle.load(f) 
X = data[:,:-6]
y = data[:,-6:]
print X.shape

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

# X_train = X_train.T

model = Sequential([
    Dense(64, input_shape=(136,)),
    Activation('sigmoid'),
    Dense(32),
    Activation('relu'),
    Dense(6),
    Activation('softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_data = (X_test, y_test))
print model.predict(np.reshape(X_train[0], (1, 136)))
# model.save('model.h5')
with open('model1.pickle','wb') as f:
	pickle.dump(model)
tf.keras.backend.clear_session()
print 'model saved!'
