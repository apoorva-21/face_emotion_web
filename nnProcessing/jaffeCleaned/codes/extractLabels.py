import numpy as np 
import pickle

file = './labels.txt'
# HAP SAD SUR ANG DIS FEA PIC
text = []
with open(file, 'rb') as f:
	text = f.read()
	text = text.split('\n')
	text = text[2:len(text)-1] #remove the header and the last \n

ratings= []
filenames= []
outFile = open('./labelsFinal.csv', 'wb')

for line in text:
	label = line.split(' ')
	filename = "./clean/" + label[-1] + ".tiff"    #only the last column
	rating = label[1:-1]                            #2nd col to last col, omit index and filename
	for i in range(len(rating)):
		rating[i] = float(rating[i])
	label = rating
	ratings.append(label)
	filenames.append(filename)

ratings = np.array(ratings)
filenames = np.array(filenames)

with open('./pickles/ratings.pickle','wb') as p:
	pickle.dump(ratings, p)

with open('./pickles/filenames.pickle','wb') as p:
	pickle.dump(filenames, p)
