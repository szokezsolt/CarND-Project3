import csv
import cv2
import numpy as np
import sklearn
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split

samples = []
steer_limit = 0.025

with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
    	if abs(float(sample[3])) > steer_limit:
        	samples.append(sample)

images = []
angles = []
augmented_images = []
augmented_angles = []
correction = 0.15
  
for sample in samples:            
	for camera_idx in range(3):
		name = './IMG/'+sample[camera_idx].split('/')[-1]
		image = cv2.imread(name)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		images.append(image)	
		angle = float(sample[3])
		angles.append(angle)
		angles.append(angle+correction)
		angles.append(angle-correction)

for image, angle in zip(images, angles):
	augmented_images.append(image)
	augmented_angles.append(angle)
	flipped_image = cv2.flip(image, 1)
	flipped_angle = float(angle * -1.0)
	augmented_images.append(flipped_image)
	augmented_angles.append(flipped_angle) 

X = np.array(augmented_images)
y = np.array(augmented_angles)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,(5,5),strides=(2,2)))
model.add(Activation('elu'))
model.add(Dropout(0.4))
model.add(Convolution2D(36,(5,5),strides=(2,2)))
model.add(Activation('elu'))
model.add(Dropout(0.4))
model.add(Convolution2D(48,(5,5)))
model.add(Activation('elu'))
model.add(Dropout(0.4))
model.add(Convolution2D(64,(3,3)))
model.add(Activation('elu'))
model.add(Dropout(0.4))
model.add(Convolution2D(64,(3,3)))
model.add(Activation('elu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=3, verbose=1, validation_split=0.25, shuffle=True)

model.save('model.h5')








