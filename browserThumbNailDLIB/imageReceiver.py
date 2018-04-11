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
app = Flask(__name__)
CORS(app)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
flag = False

def detectFaces(img):
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        shapeNP = face_utils.shape_to_np(shape)
        for x, y in shapeNP:
            cv2.circle(img,(x,y),2,(255,0,0),-1)
    return img


@app.route('/uploadImage/', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def get_image():
    image_b64 = request.json['image']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64).decode('base64')
    image_PIL = Image.open(cStringIO.StringIO(image_data))
    image_np = np.array(image_PIL)
    image_np = image_np[:,:,:3]
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    print 'Image received: {}'.format(image_np.shape)
    image_np = detectFaces(image_np)
    cv2.imshow("frame", image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    response = jsonify({'msg':'Received the Image'})
    return response

if __name__ == '__main__':
	app.run(port = 8080, debug = True)
