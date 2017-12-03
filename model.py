#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:08:03 2017

@author: szokezsolt
"""

import csv
import cv2
import math
import numpy as np
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam

csv_file = './driving_log.csv'
out_file = './model.h5'
epochs = 1
learning_rate = 0.001
angle_correction = 0.25
trans_max = 50
steer_limit = 0.05
batch_size = 64

def read_data(csv_file):
    with open(csv_file) as file:
        X = []
        y = []
        reader = csv.reader(file)
        for sample in reader:
            if abs(float(sample[3])) > steer_limit:
                img_name_center = sample[0].strip()
                img_name_left = sample[1].strip()
                img_name_right = sample[2].strip()
                angle = float(sample[3])
                X.append([img_name_center, img_name_left, img_name_right])
                y.append(angle)
    return X, y

def normalize(img):
    return img / 255.0 - 0.5

def augmentation(img, steering):
    trans = np.random.uniform(-trans_max, trans_max)
    steering = steering + (trans / trans_max) * angle_correction
    matrix = np.float32([[1,0,trans],[0,1,0]])
    img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    if np.random.randint(2):
        img = cv2.flip(img,1)
        steering = -steering
    return img, steering

def create_network(input_shape, learning_rate=0.001): 
    model = Sequential()
    model.add(Lambda(normalize, input_shape=input_shape, output_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,(5,5),strides=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(36,(5,5),strides=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(48,(5,5),strides=(1,1)))
    model.add(ELU())
    model.add(Convolution2D(64,(3,3),strides=(1,1)))
    model.add(ELU())
    model.add(Convolution2D(64,(3,3),strides=(1,1)))
    model.add(ELU())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(ELU())
    model.add(Dense(100))
    model.add(Dropout(0.25))
    model.add(ELU())
    model.add(Dense(50))
    model.add(Dropout(0.25))
    model.add(ELU())
    model.add(Dense(10))
    model.add(Dropout(0.25))
    model.add(ELU())
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model
    
def training_generator(X, y, batch_size):
    angle_corrections = [0.0, angle_correction, -angle_correction]
    while 1:
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            idx = np.random.randint(len(y))
            idx_img = np.random.randint(len(angle_corrections))
            x_i = mpimg.imread(X[idx][idx_img].strip())
            y_i = y[idx] + angle_corrections[idx_img]
            x_i, y_i = augmentation(x_i, y_i)
            x_out.append(x_i)
            y_out.append(y_i)
        yield (np.array(x_out), np.array(y_out))

def validation_generator(X, y):
    while 1:
        for i in range(len(y)):
            x_out = mpimg.imread(X[i][0])
            y_out = np.array([[y[i]]])
            x_out = x_out[None,:,:,:]
        yield x_out, y_out
        
def correct_training_samples(X, batch_size):
    return int(math.ceil(float(X) / float(batch_size)) * batch_size)
    
def train_network(model, epochs, X_train, y_train):
    train_samples = 5 * correct_training_samples(len(y_train), batch_size)
    valid_samples = len(y_train)
    train_gen = training_generator(X_train, y_train, batch_size)
    valid_gen = validation_generator(X_train, y_train)
    model.fit_generator(generator=train_gen, 
                        validation_data=valid_gen, 
                        validation_steps=valid_samples, 
                        epochs=epochs, 
                        verbose=1,
                        steps_per_epoch=train_samples, 
                        use_multiprocessing=False)
    model.save(out_file)
    
def main():
    X, y = read_data(csv_file)
    model = create_network((160,320,3), learning_rate)
    train_network(model, epochs, X, y)
    
if __name__ == '__main__':
    main()  