import numpy as np
from PIL import Image
import base64
import re
import cStringIO
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import cv2
from imutils import face_utils
import dlib
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


app = Flask(__name__)
CORS(app)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
flag = False

emotions = ['HAPPY', 'SAD', 'SURPRISED', 'ANGRY', 'DISGUSTED', 'FEARFUL']

def trainModel():
    global X, y
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
    return model



def makePred(vector):
    model = trainModel()
    return model.predict(vector)

def detectFaces(img):
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        shapeNP = face_utils.shape_to_np(shape)
        for x, y in shapeNP:
            cv2.circle(img,(x,y),2,(255,0,0),-1)
    return img

def detectFacesGetVector(img):
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        shapeNP = face_utils.shape_to_np(shape)
        faceCentre = shapeNP[30]        
        hNormPoint1 = shapeNP[0]
        hNormPoint2 = shapeNP[16]       
        vNormPoint1 = shapeNP[21]
        vNormPoint2 = shapeNP[8]
        hNormDist = hNormPoint2[0] - hNormPoint1[0]
        vNormDist = vNormPoint2[1] - vNormPoint1[1]
        #absolute X distances from central point normalized
        faceVectorX = np.array(np.abs(shapeNP[:,0] - faceCentre[0]), np.float32)/hNormDist
        #absolute Y distances from central point normalized
        faceVectorY = np.array(np.abs(shapeNP[:,1] - faceCentre[1]), np.float32)/vNormDist
        #append the distances to get a 136-long vector per face
        faceVector = np.hstack([faceVectorX, faceVectorY])
        for x, y in shapeNP:
            cv2.circle(img,(x,y),2,(255,0,0),-1)
        return img,faceVector

@app.route('/uploadImage/', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def get_image():
    global model
    image_b64 = request.json['image']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64).decode('base64')
    image_PIL = Image.open(cStringIO.StringIO(image_data))
    image_np = np.array(image_PIL)
    image_np = image_np[:,:,:3]
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    print 'Image received: {}'.format(image_np.shape)
    # image_np = detectFaces(image_np)
    image_np, faceVector = detectFacesGetVector(image_np)
    print faceVector.shape
    cv2.imshow("frame", image_np)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    predictionVec = makePred(np.reshape(faceVector, (1, 136)))
    prediction = np.argmax(predictionVec)
    response = jsonify({'emotion':emotions[prediction]})
    return response
    # return 'ok'

if __name__ == '__main__':
	app.run(port = 8080, debug = True)
