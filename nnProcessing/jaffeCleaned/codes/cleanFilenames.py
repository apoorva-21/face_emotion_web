import cv2
import numpy as np
import os

CLEAN_DIR = './clean' 
for path in os.listdir('./jaffe'):
	dirtyPath = os.path.join('./jaffe',path)
	img = cv2.imread(dirtyPath)
	cleanPath = path.split('.')[:2]
	cleanPath = os.path.join(CLEAN_DIR, cleanPath[0] + '-' + cleanPath[1] + '.tiff')
	print cleanPath
	cv2.imwrite(cleanPath, img)