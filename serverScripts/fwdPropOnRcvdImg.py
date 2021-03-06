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
from keras.models import load_model
from keras.models import model_from_json
import keras
app = Flask(__name__)
CORS(app)
MODEL_NAME = '../trainedModels/model1500.h5'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../facialLandmarkData/shape_predictor_68_face_landmarks.dat')
flag = False

emotions = ['HAPPY', 'SAD', 'SURPRISED', 'ANGRY', 'DISGUSTED', 'FEARFUL']
#model = load_model(MODEL_NAME)


# load json and create model
json_file = open('../trainedModels/model1500.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../trainedModels/model1500.h5")
print("Loaded model from disk")
loaded_model._make_predict_function()
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
    global loaded_model
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
    predictionArray = loaded_model.predict(np.reshape(faceVector, (1, 136)))
    #oneHotPred = model.predict(np.reshape(faceVector, (1,136)))
    prediction = np.argmax(predictionArray)
    print emotions[prediction]
    response = jsonify({'emotion':emotions[prediction]})
    return response
if __name__ == '__main__':
	app.run(port = 8080, debug = True)
