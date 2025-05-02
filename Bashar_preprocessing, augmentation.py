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