**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing a lap of driving the car autonomously

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. I prefer Uncle Bob's clean code approach: I never use comments in self-explanatory codes.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes with depths between 24 and 64 (model.py lines 60-86). Before this the input is normalized and cropped. The model includes ELU layers to introduce nonlinearity. After a max pooling and a flattening layer, 4 fully connected layer finalizes the structure, using 100-50-10-1 nodes, including dropout layers as well. The total number of parameters is 789'819.

####2. Attempts to reduce overfitting in the model 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Image processing includes augmentation of the available images by a randomized translation, selection of the three available cameras with steering correction, flip, plus of course normalization & cropping.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

I used Udacity's training data to keep the vehicle driving on the road. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use NVIDIA's CNN architecture, as a basis, and implement some adjustment to fit the use case of this project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to generalize its behaviour: flip, translate images and use all 3 cameras, randomly 

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track infinitely without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 60-86) consisted of a convolution neural network. Here is the summary of the network: 

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
elu_1 (ELU)                  (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
elu_2 (ELU)                  (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 73, 48)        43248     
_________________________________________________________________
elu_3 (ELU)                  (None, 10, 73, 48)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 71, 64)         27712     
_________________________________________________________________
elu_4 (ELU)                  (None, 8, 71, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 6, 69, 64)         36928     
_________________________________________________________________
elu_5 (ELU)                  (None, 6, 69, 64)         0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 3, 34, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6528)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6528)              0         
_________________________________________________________________
elu_6 (ELU)                  (None, 6528)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               652900    
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
elu_7 (ELU)                  (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
elu_8 (ELU)                  (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_4 (Dropout)          (None, 10)                0         
_________________________________________________________________
elu_9 (ELU)                  (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 789,819
Trainable params: 789,819
Non-trainable params: 0

####3. Creation of the Training Set & Training Process

I used Udacity's default training data. During the collection process, I excluded the samples with a steering angle less, than 0.05 to avoid tending to go straight.

I used separate generators to create training and validation data to avoid loading all images into the memory. Validation generator uses no image augmentation. A separate function (model.py line 114-115) helps to determine the correct number of training samples based on the batch size to avoid warnings.

I finally had 16'000 number of data points, randomly shuffled the data set and put 1/6 of the data into a validation set on the fly. 

I used this training data for training the model. I used only 1 epoch, since the model was already working well. I used an adam optimizer so that manually training the learning rate wasn't necessary.