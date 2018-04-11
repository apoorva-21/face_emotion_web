import numpy as np
import os
import cv2
import dlib
from imutils import face_utils
import pickle

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
flag = False

def getFiducialPointsData(imagePath):
	global detector, predictor
	print imagePath
	try :
		img = cv2.imread(imagePath)
		img = img.astype(np.uint8)
	except:
		return 'nf'
	dets = detector(img, 1)
	faceVector = np.zeros((136,), np.float32)
	print("{} : Number of faces detected: {}".format(imagePath,len(dets)))
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
		return faceVector


LABELS_FILE = './labels.txt'
IMAGES_DIR = './jaffeCleaned/data'
labels = []
with open(LABELS_FILE, 'rb') as f:
	dataset = f.read()
items = dataset.split('\n')
items = items[2:-1]

oneHotLabels = []
ds = []
for item in items:
	itemAsList = item.split(' ')
	label = np.array(itemAsList[1:-1], np.float32)
	oneHotLabel = list(np.zeros(label.shape, np.int16))
	oneHotLabel[np.argmax(label)] = 1
	imagePath = os.path.join(IMAGES_DIR, itemAsList[-1]) + '.tiff'
	#process image here
	dataItem = getFiducialPointsData(imagePath)
	if dataItem == 'nf' :
		print 'SKIPPED!!!!!'
	else:
		oneHotLabel = np.array(oneHotLabel)
		ds.append(np.hstack([dataItem,oneHotLabel]))
print "DATASET CREATED! Shape of Samples = {}".format(np.array(ds).shape)
with open('faceDataset.pickle', 'wb') as f:
	pickle.dump(np.array(ds), f)
print 'Saved as faceDataset.pickle!'