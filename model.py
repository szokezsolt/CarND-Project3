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
correction = 0.25
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.3)

def generator(samples, batch_size=16):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			augmented_images = []
			augmented_angles = []
            
			for batch_sample in batch_samples:
				#for camera_idx in range(3):
					#source_path = line[camera_idx]
				source_path = line[1]
				name = './IMG/'+batch_sample[0].split('/')[-1]
				image = cv2.imread(name)
				image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
				images.append(image)	
				angle = float(batch_sample[3])
				#angles.append(angle)
				angles.append(angle+correction)
				#angles.append(angle-correction)

			for image, angle in zip(images, angles):
				augmented_images.append(image)
				augmented_angles.append(angle)
				flipped_image = cv2.flip(image, 1)
				flipped_angle = float(angle * -1.0)
				augmented_images.append(flipped_image)
				augmented_angles.append(flipped_angle) 

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=1)
validation_generator = generator(validation_samples, batch_size=1)

"""
X_train, y_train = next(train_generator)
X_validation, y_validation = next(validation_generator)

n, bins, patches = plt.hist(y_train, 40, facecolor='green', alpha=0.75,label=['Training data'])
n, bins, patches = plt.hist(y_validation, 40, facecolor='red', alpha=0.75,label=['Validation data'])
plt.xlabel('Steering value')
plt.ylabel('Number of images')
plt.title('Histogram of steering values')
plt.legend()
plt.savefig('histogram.png', bbox_inches='tight')
"""

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,(5,5),strides=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(36,(5,5),strides=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(48,(5,5)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)

model.save('model.h5')

