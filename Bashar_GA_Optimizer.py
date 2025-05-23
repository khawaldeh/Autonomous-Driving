# -*- coding: utf-8 -*-
"""Bashar_GA_BBL_Optimizer.ipynb

!apt install unzip

!unzip /content/Dataset -d /content/sample_data

!pip3 install imgaug # first setp of creating image data generator

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from keras.callbacks import TensorBoard
import cv2
import pandas as pd
import ntpath
import random
from keras.models import load_model
import copy
from sklearn.preprocessing import MinMaxScaler
random.seed(1) #change it each run

datadir = '/content/sample_data/Dataset'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data= pd.read_csv(os.path.join (datadir, 'driving_log.csv'), names = columns)
#pd.set_option('display.max_colwidth', -1) # make the table shows all data
data.head()

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail

data ['center'] = data ['center'].apply(path_leaf)
data ['left'] = data ['left'].apply(path_leaf)
data ['right'] = data ['right'].apply(path_leaf)
data.head()

num_bins = 25 # we will use it to divide the steering into 25 intervals
samples_per_bin = 200
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))

#Blancing data
print('total data:', len(data))

remove_list = []

for j in range(num_bins):
  list_ = []
  for i in range(len(data['steering'])): # to itrate through the range of all the steering data
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]: # isolate the steering angles belong to the current bin being itrated through
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]  #isolate the indices that go beyond the threshold
  remove_list.extend(list_)


print ('removed:', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print ('remaining:', len(data))


hist,_ = np.histogram(data['steering'], (num_bins))
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']),np.max(data['steering'])),(samples_per_bin,samples_per_bin))

#training & validation split
print(data.iloc[1])
def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]   # ilco allows us to perform a selection on a row of data from our data frame based on the specified index
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))
    # left image append
    image_path.append(os.path.join(datadir, left.strip()))
    steering.append(float(indexed_data[3])+0.15)
    # right image append
    image_path.append(os.path.join(datadir, right.strip()))
    steering.append(float(indexed_data[3])-0.15)
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir + '/IMG', data)

x_train, x_valid, y_train,  y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('traning samples:{}\nValid samples: {}'.format(len(x_train), len(x_valid)))

fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title('Validation set')

# creating augmentation technique as individual function
# augmintation is the process of creating new data images using existing dataset which is done by transformation the image
def zoom(image): # will zoom in the image and improve the feature extraction
  zoom = iaa.Affine(scale=(1, 1.3))
  image = zoom.augment_image(image)
  return image #this function very usful for augmentation data

# to visulize augmentation
image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom (original_image)

fig, axs = plt.subplots(1, 2 , figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')

axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Image')

def pan(image):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1 , 0.1), "y" : (-0.1 , 0.1)})
  image = pan.augment_image(image)
  return (image)

# to visulize augmentation // pan
image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
paned_image = pan(original_image)

fig, axs = plt.subplots(1, 2 , figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')

axs[1].imshow(paned_image)
axs[1].set_title('Paned Image')

def img_random_brightness(image):    #multiplies all the pixle itinsties thus any pixel multiblied by value less than 1 will become darker
  brightness = iaa.Multiply((0.2 , 1.2))
  image = brightness.augment_image(image)
  return (image)

#to visulize augmentation // brightness
image = image_paths [random.randint(0, 1000)]
original_image = mpimg.imread(image)
brightness_altered_image = img_random_brightness(original_image)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')

axs[1].imshow(brightness_altered_image)
axs[1].set_title('brightness alterrd Image')

def img_random_flip(image, steering_angle):
  image = cv2.flip(image , 1)
  steering_angle = -steering_angle
  return image, steering_angle

#to visulize augmentation // flip
random_index= random.randint(0, 1000)
image = image_paths [random_index]
steering_angle = steerings[random_index]

original_image = mpimg.imread(image)
flipped_image, flipped_steering_angle = img_random_flip (original_image, steering_angle)

fig, axs = plt.subplots(1, 2 , figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image -' + 'steering angle:' + str(steering_angle))

axs[1].imshow(flipped_image)
axs[1].set_title('flipped image -'+ 'steering angle:' + str(flipped_steering_angle))

def random_augment (image, steering_angle):
  image = mpimg.imread(image)
  if np.random.rand() < 0.5:  # 50% of the time return value between 0 and 1
    image = pan (image)
  if np.random.rand() < 0.5:
    image = zoom (image)
  if np.random.rand() < 0.5:
    image = img_random_brightness (image)
  if np.random.rand() < 0.5:
    image, steering_angle = img_random_flip(image, steering_angle)
  return image, steering_angle

ncol = 2
nrow = 10

fig, axs = plt.subplots(nrow, ncol , figsize=(15, 50))
fig.tight_layout()

for i in range(10):
  randnum = random.randint (0 , len(image_paths)-1)
  random_image = image_paths[randnum]
  random_steering = steerings[randnum]

  original_image = mpimg.imread(random_image)
  augmented_image, steering = random_augment(random_image, random_steering)

  axs[i][0].imshow(original_image)
  axs[i][0].set_title("original_image")

  axs[i][1].imshow(augmented_image)
  axs[i][1].set_title("augmented_image")

def img_preprocess (img):
  img = img[60:135, : , :]  # crop the image which will allow our model to foucus on the more importent features on the image such as the lane lines and the borders
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) # Y represent the brightness and UV represent chromium which add colors to the image / expirtis recomended that this color space is moe effective for use in training
  img = cv2.GaussianBlur (img, (3,3), 0) # (3,3) the size of hte kernal of the gaussian
  img = cv2.resize(img, (200, 66)) # decrease the image size to match the image size of the input image used by the nivada model artiticture
  img = img/255 # for normalization
  return img

image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(original_image)

fig, axs = plt.subplots (1, 2, figsize=(15,10)) #one row and two column
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original image')
axs[1].imshow(preprocessed_image)
axs[1].set_title('preprocessed image')

# coded the generator
def batch_generator (image_paths, steering_ang, batch_size, istraining):
  # yelid keyword still returns desired output of the batch_generator it also save all initialized value insied the batch generator


  while True:
    batch_img = []
    batch_steering = []

    for i in range (batch_size):
      random_index = random.randint(0, len(image_paths) -1)

      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])

      else:
        im = mpimg.imread (image_paths[random_index])
        steering = steering_ang[random_index]

      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asanyarray(batch_img), np.asanyarray(batch_steering))

# fit the generator
x_train_gen, y_train_gen = next(batch_generator(x_train, y_train, 1, 1)) # number 1 is present that call the generator to generate one image.
x_valid_gen, y_valid_gen = next(batch_generator(x_valid, y_valid, 1, 0))

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(x_train_gen[0])
axs[0].set_title('Training Image')

axs[1].imshow(x_valid_gen[0])
axs[1].set_title('Validation Image')

# if you are using "supsample" with keras via the Tensoflow 2.0API has been changed to strides and intead of (24, 5, 5 ...) it will be (24, kernel_size=( 5, 5)...)
# create nivida model for training
def nvidia_model():
  model = Sequential()  #input layer
  model.add(Convolution2D (24, kernel_size=( 5, 2), strides=(2,2), input_shape = (66,200,3), activation='elu'))# wechange 'relu' to 'elu' to avoid dying problem
  model.add(Convolution2D (36, kernel_size=( 5, 2),  strides=(2,2), activation='elu'))# with 'elu' we elimnait the overfitting completly
  model.add(Convolution2D (48, kernel_size=( 5, 2), strides=(2,2), activation='elu'))
  model.add(Convolution2D (64, kernel_size=( 3, 3),  activation='elu')) # 'relu' alwase has chance to fix the weight to decrease it's error


  model.add(Convolution2D (64, kernel_size=( 3, 3),  activation='elu')) # 'relu' we change it to 'sigmoid' in the more complex neural network
  #model.add(Dropout(0.5))

  model.add(Flatten())

  model.add(Dense(100, activation='elu'))
  #model.add(Dropout(0.5))


  model.add(Dense(50, activation='elu'))
  #model.add(Dropout(0.5))


  model.add(Dense(10, activation='elu'))
  #model.add(Dropout(0.5))

  model.add(Dense(1))

  # Compile with Adam optimizer
  model.compile(loss='MSE', optimizer = 'adam')

  return model

# Train and save 32 models
for i in range(32):
    model = nvidia_model()
    history = model.fit(batch_generator(x_train,  y_train, 100, 1 ),
                                steps_per_epoch=300,
                                epochs = 50,
                                validation_data= batch_generator(x_valid, y_valid, 50, 0),
                                validation_steps=200,
                                verbose=1,
                                shuffle=1,)

    model.save('/content/Models/model' + str(i) + '.h5')

# upload the pre-traind models to the content
E = []
for i in range(32):
  C = keras.models.load_model('/content/Models/model' + str(i) + '.h5')
  E1 = C.evaluate(batch_generator(x_valid,  y_valid, 100, 1 ), verbose=1, steps=50)
  E.append(E1[0])
print(E)

def Crossover(X1,Y1):
  X = X1.weights
  Y = Y1.weights
  u = random.randint(0, 17)
  #print(u)

  if u == 0:
    q = [random.randint(0,4),random.randint(0,1),random.randint(0,2),random.randint(0,23)]
    z1 = X[u][q[0]][q[1]][q[2]]
    z2 = Y[u][q[0]][q[1]][q[2]]
    indices1 = [[q[0],q[1],q[2]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 1:
    q = [random.randint(0,23)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 2:
    q = [random.randint(0,4),random.randint(0,1),random.randint(0,23),random.randint(0,35)]
    z1 = X[u][q[0]][q[1]][q[2]]
    z2 = Y[u][q[0]][q[1]][q[2]]
    indices1 = [[q[0],q[1],q[2]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 3:
    q = [random.randint(0,35)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 4:
    q = [random.randint(0,4),random.randint(0,1),random.randint(0,35), random.randint(0,47)]
    z1 = X[u][q[0]][q[1]][q[2]]
    z2 = Y[u][q[0]][q[1]][q[2]]
    indices1 = [[q[0],q[1],q[2]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 5:
    q = [random.randint(0,47)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 6:
    q = [random.randint(0,2),random.randint(0,2),random.randint(0,47),random.randint(0,63)]
    z1 = X[u][q[0]][q[1]][q[2]]
    z2 = Y[u][q[0]][q[1]][q[2]]
    indices1 = [[q[0],q[1],q[2]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 7:
    q = [random.randint(0,63)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 8:
    q = [random.randint(0,2),random.randint(0,2),random.randint(0,63),random.randint(0,63)]
    z1 = X[u][q[0]][q[1]][q[2]]
    z2 = Y[u][q[0]][q[1]][q[2]]
    indices1 = [[q[0],q[1],q[2]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 9:
    q = [random.randint(0,63)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 10:
    q = [random.randint(0,1343),random.randint(0,99)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 11:
    q = [random.randint(0,99)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 12:
    q = [random.randint(0,99),random.randint(0,49)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 13:
    q = [random.randint(0,49)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 14:
    q = [random.randint(0,49),random.randint(0,9)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 15:
    q = [random.randint(0,9)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 16:
    q = [random.randint(0,9)]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))
  if u == 17:
    q = [0]
    z1 = X[u][q[0]]
    z2 = Y[u][q[0]]
    indices1 = [[q[0]]]
    X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices1, [z2]))
    Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices1, [z1]))

  return(X1, Y1)

def Mutation (X1,Y1):
  X = X1.weights
  Y = Y1.weights

  for k in range(10):
    mr = random.random()
    if mr < 0.2:
      u = random.randint(0, 17)
      #print(u)
      if u == 0:
        for i in range (2):
          t = [random.randint(0,4),random.randint(0,1),random.randint(0,2),random.randint(0,23)]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0],t[1],t[2],t[3]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 1:
        for i in range (2):
          t = [random.randint(0,23)]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 2:
        for i in range (2):
          t = [random.randint(0,4),random.randint(0,1),random.randint(0,23),random.randint(0,35)]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0],t[1],t[2],t[3]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 3:
        for i in range (2):
          t = [random.randint(0,35)]
          #zm1 = 1 - X[u][t[0]]
          #zm2 = 1 - Y[u][t[0]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 4:
        for i in range (2):
          t = [random.randint(0,4),random.randint(0,1),random.randint(0,35), random.randint(0,47)]
          #zm1 = 1 - X[u][t[0]][t[1]][t[2]][t[3]]
          #zm2 = 1 - Y[u][t[0]][t[1]][t[2]][t[3]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0],t[1],t[2],t[3]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 5:
        for i in range (2):
          t = [random.randint(0,47)]
          #zm1 = 1 - X[u][t[0]]
          #zm2 = 1 - Y[u][t[0]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 6:
        for i in range (2):
          t = [random.randint(0,2),random.randint(0,2),random.randint(0,47),random.randint(0,63)]
          #zm1 = 1 - X[u][t[0]][t[1]][t[2]][t[3]]
          #zm2 = 1 - Y[u][t[0]][t[1]][t[2]][t[3]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0],t[1],t[2],t[3]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 7:
        for i in range (2):
          t = [random.randint(0,63)]
          zm1 = random.random()
          zm2 = random.random()
          #zm1 = 1 - X[u][t[0]]
          #zm2 = 1 - Y[u][t[0]]
          indices2 = [[t[0]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 8:
        for i in range (2):
          t = [random.randint(0,2),random.randint(0,2),random.randint(0,63),random.randint(0,63)]
          #zm1 = 1 - X[u][t[0]][t[1]][t[2]][t[3]]
          #zm2 = 1 - Y[u][t[0]][t[1]][t[2]][t[3]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0],t[1],t[2],t[3]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 9:
        for i in range (2):
          t = [random.randint(0,63)]
          #zm1 = 1 - X[u][t[0]]
          #zm2 = 1 - Y[u][t[0]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 10:
        for i in range (2):
          t = [random.randint(0,1343),random.randint(0,99)]
          #zm1 = 1 - X[u][t[0]][t[1]]
          #zm2 = 1 - Y[u][t[0]][t[1]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0],t[1]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 11:
        for i in range (2):
          t = [random.randint(0,63)]
          #zm1 = 1 - X[u][t[0]]
          #zm2 = 1 - Y[u][t[0]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 12:
        for i in range (2):
          t = [random.randint(0,99),random.randint(0,49)]
          #zm1 = 1 - X[u][t[0]][t[1]]
          #zm2 = 1 - Y[u][t[0]][t[1]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0],t[1]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 13:
        for i in range (2):
          t = [random.randint(0,49)]
          #zm1 = 1 - X[u][t[0]]
          #zm2 = 1 - Y[u][t[0]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 14:
        for i in range (2):
          t = [random.randint(0,49),random.randint(0,9)]
          #zm1 = 1 - X[u][t[0]][t[1]]
          #zm2 = 1 - Y[u][t[0]][t[1]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0],t[1]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 15:
        for i in range (2):
          t = [random.randint(0,9)]
          #zm1 = 1 - X[u][t[0]]
          #zm2 = 1 - Y[u][t[0]]
          zm1 = random.random()
          zm2 = random.random()
          indices2 = [[t[0]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 16:
        for i in range (2):
          t = [random.randint(0,9)]
          #zm1 = 1 - X[u][t[0]]
          #zm2 = 1 - Y[u][t[0]]
          zm1 = [random.random()]
          zm2 = [random.random()]
          indices2 = [[t[0]]]
          X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
          Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))
      if u == 17:
        t = [0]
        #zm1 = 1 - X[u][q[0]]
        #zm2 = 1 - Y[u][q[0]]
        zm1 = random.random()
        zm2 = random.random()
        indices2 = [[t[0]]]
        X1.weights[u].assign(tf.tensor_scatter_nd_update(X1.weights[u], indices2, [zm1]))
        Y1.weights[u].assign(tf.tensor_scatter_nd_update(Y1.weights[u], indices2, [zm2]))

  return(X1, Y1)

# to the folder empity
def empty(f):
  import os, shutil
  folder = f
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_min_value(z):
  L = copy.deepcopy(z)
  m_v = []
  for i in range(16):
    mv = np.argmin(L)
    #print (mv)
    L[mv] = 100
    m_v.append(mv)
  #print (L)
  return m_v

for g in range(50):
  v = get_min_value(E)
  print(v)
  Ep = []
  Ec = []

  # Parents Selection
  for i in range(16):
    #print('i', i)
    X = keras.models.load_model('/content/Models/model' + str(v[i]) + '.h5')
    X.save('/content/Parents/model' + str(i) + '.h5')
    #print(W[v[i]])
    Ep.append(E[v[i]])
  #print(Ep)

  # Chidren
  for i in range(0,16,2):
    csum1 = 0
    csum2 = 0
    msum1 = 0
    msum2 = 0
    X = keras.models.load_model('/content/Parents/model' + str(i) + '.h5')
    Y = keras.models.load_model('/content/Parents/model' + str(i+1) + '.h5')
    time.sleep(0.0001)
    X.save('/content/Xn/modelXn' + str(i) + '.h5')
    Y.save('/content/Xn/modelXn' + str(i+1) + '.h5')
    Xn= keras.models.load_model('/content/Xn/modelXn' + str(i) + '.h5')
    Yn= keras.models.load_model('/content/Xn/modelXn' + str(i+1) + '.h5')

    C1, C2 = Crossover(Xn,Yn)
    M1, M2 = Mutation(C1, C2)
    M1.save('/content/Childs/model' + str(i) + '.h5')
    M2.save('/content/Childs/model' + str(i+1) + '.h5')

  # Evaluation for Children
  for c in range(16):
    Ch = keras.models.load_model('/content/Childs/model' + str(c) + '.h5')
    sum = 0
    print('ch', c)

    for j in range (5):
      U1 = Ch.evaluate(batch_generator(x_train,  y_train, 50, 1 ), verbose=1, steps=50)
      sum += U1
      avge = sum / (j+1)

      if U1 > 0.17:
        print("Skipping this iteration.")
        break
    print(avge)
    Ec.append(avge)
  #print(Ec)
  #print(np.amin(Ec))

  # Replacment
  for i in range(16):
    mv1 = np.argmax(E)
    mv2 = np.argmin(Ec)
    #print('mv1',mv1, 'mv2', mv2 )
    if Ec[mv2] < E[mv1]:
      #print('E[mv1]',E[mv1], 'Ec[mv2]', Ec[mv2] )
      E[mv1] = Ec[mv2]
      print(Ec[mv2])
      #print('E[mv1]',E[mv1], 'Ec[mv2]', Ec[mv2] )
      os.remove('/content/Models/model' + str(mv1) + '.h5')
      time.sleep(0.0001)
      t1 = keras.models.load_model('/content/Childs/model' + str(mv2) + '.h5')
      t1.save('/content/Models/model' + str(mv1) + '.h5')
      Ec[mv2] = 100

  print(Ec)
  print('g', g)
  print(E)
  print(np.amin(E))
  print(np.average(E))
  print(np.std(E))
  empty('/content/Childs')
  empty('/content/Parents')
  empty('/content/Xn')

from google.colab import files
!zip -r /content/models.zip /content/Models

!cp -r '/content/models.zip' /content/drive/MyDrive/
