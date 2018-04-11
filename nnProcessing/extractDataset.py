import dlib
import cv2
from imutils.video import WebcamVideoStream
from imutils import face_utils
import numpy as np
import pickle
# N_SAMPLES_TO_COLLECT = 100

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
flag = False
#cap = WebcamVideoStream(src = 0).start()

label_dict = {'Happy':0, 'Sad':0}

collecting = 'Happy'

def getFPDistanceVector(points, centralPoint):
	return np.sum((points - centralPoint)**2, axis = 1)**0.5

def getXYDistancesSeparately(points, centralPoint):
	return np.ravel(np.abs(points - centralPoint)) 
filenames = []
with open('./pickles/filenames.pickle','rb') as p:
	filenames = pickle.load(p)
print filenames.shape
print filenames[0]

radialSamples = []
XYSamples = []
for i in range(filenames.shape[0]):
    img = cv2.imread(filenames[i])
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    img = img.astype(np.uint8)
    dets = detector(img, 1)
    cv2.imshow('raw',img)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        shapeNP = face_utils.shape_to_np(shape)
        
        faceCentre = shapeNP[30] 
        radialDistances = getFPDistanceVector(shapeNP, faceCentre)
        XYDistances = getXYDistancesSeparately(shapeNP, faceCentre)

        radialSample = np.hstack([radialDistances,label_dict[collecting]])
        radialSamples.append(radialSample)
        
        XYSample = np.hstack([XYDistances,label_dict[collecting]])
        XYSamples.append(XYSample)
        
        for i in range(shapeNP.shape[0]):
        	# cv2.circle(img,(shapeNP[i,0], shapeNP[i,1]),2,(255,0,0),-1)
        	cv2.line(img, (faceCentre[0], faceCentre[1]), (shapeNP[i,0], shapeNP[i,1]), (0,255,255), 1)
        cv2.imshow('img',img)

# Draw the face landmarks on the screen.

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

radialSamples = np.array(radialSamples, np.float32)
print radialSamples.shape

XYSamples = np.array(XYSamples, np.float32)
print XYSamples.shape

with open('./pickles/data_radial.pickle', 'wb') as f:
	pickle.dump(radialSamples, f)


with open('./pickles/data_XY.pickle', 'wb') as f:
	pickle.dump(XYSamples, f)

print('Pickle file of {} samples for {} created successfully!'.format(N_SAMPLES_TO_COLLECT, collecting))

exit()
