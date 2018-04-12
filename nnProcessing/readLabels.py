import numpy as np
import os
import cv2

LABELS_FILE = './dataset/labels.txt'
IMAGES_DIR = './jaffeCleaned/data'
labels = []
with open(LABELS_FILE, 'rb') as f:
	dataset = f.read()
items = dataset.split('\n')
items = items[2:-1]

oneHotLabels = []
for item in items:
	itemAsList = item.split(' ')
	label = np.array(itemAsList[1:-1], np.float32)
	oneHotLabel = list(np.zeros(label.shape, np.int16))
	oneHotLabel[np.argmax(label)] = 1
	oneHotLabels.append(oneHotLabel)
	imagePath = os.path.join(IMAGES_DIR, itemAsList[-1])
	print oneHotLabel, imagePath
