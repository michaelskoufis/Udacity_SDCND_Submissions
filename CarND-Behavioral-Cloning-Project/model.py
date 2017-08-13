import os
import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.utils import shuffle
from keras.backend import tf as ktf
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Activation, Dropout, Lambda, Convolution2D, MaxPooling2D, Cropping2D

# Steer angle shifts for the center, left and right images respectively
delta_steering = [0.0,0.2,-0.2]

def resize_image( image ):
    # resize the image
    resized = cv2.resize( image, ( 200, 161 ), interpolation = cv2.INTER_AREA )
    resized_yuv = cv2.cvtColor( resized, cv2.COLOR_BGR2YUV )
    return resized_yuv

# Read the log file with the paths to the images along with steer angles and other information
samples = []
with open( './data_udacity/driving_log.csv' ) as csvfile:
    reader = csv.reader( csvfile )
    for line in reader:
        samples.append( line )

# Remove the label line (1st line in csv file)
samples = samples[1:]

# Limit the number of small steering angles to avoid biasing the model drive on a straight line
samp = []
ang  = []
counts = 0
for s in samples:
  a = np.float( s[ 3 ] )
  if a > -0.001 and a < 0.001:
    counts = counts + 1
    if counts < 150:
      ang.append( a )
      samp.append( s )
  else:
      ang.append( a )
      samp.append( s )
plt.hist(ang,200)
plt.show()
samples.clear()
samples = samp

# Split the set into a training set and validation set
train_samples, validation_samples = train_test_split( samples, test_size = 0.2 )

# Use a generator for a more efficient memory usage
def generator( samples, batch_size = 16 ):
    num_samples = len( samples )
    while 1: # Loop forever so the generator never terminates
        shuffle( samples )
        for offset in range( 0, num_samples, batch_size ):
            batch_samples = samples[ offset:offset+batch_size ]
            
            augmented_images = []
            augmented_angles = []
    
            # Get center, left and right images along with their flipped versions
            for batch_sample in batch_samples:
                for i in range( 3 ):
                  # Read the image and the corresponding angle measurement
                  name = './data_udacity/IMG/' + batch_sample[ i ].split('/')[ -1 ]
                  image = cv2.imread( name )
                  rescaled = resize_image( image )
                  image = np.array([])
                  image = rescaled
                  angle = np.float( batch_sample[ 3 ] ) + np.float( delta_steering[ i ] )
                  
                  # Flip the data horizontally
                  image_flipped = np.fliplr( image )
                  angle_flipped = -angle
                                            
                  # Append images and measurements
                  augmented_images.append( image )
                  augmented_images.append( image_flipped )
                  augmented_angles.append( angle )
                  augmented_angles.append( angle_flipped )
        
            # Trim image to only see section with road
            X_train = np.array( augmented_images )
            y_train = np.array( augmented_angles )
            yield sklearn.utils.shuffle( X_train, y_train )

# Compile and train the model using the generator function
train_generator = generator( train_samples, batch_size = 16 )
validation_generator = generator( validation_samples, batch_size = 16 )

# Implement the neural network
model = Sequential()
model.add( Lambda( lambda x: ( x / 255.0 ) - 0.5, input_shape = ( 161, 200, 3 ) ) )
model.add( Cropping2D( cropping = ( (70,26), (0,0) ) ) )
model.add( Convolution2D( 24, 5, 5, subsample=(2,2), activation = 'elu') )
model.add( Convolution2D( 36, 5, 5, subsample=(2,2), activation = 'elu') )
model.add( Convolution2D( 48, 5, 5, subsample=(2,2), activation = 'elu') )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'elu') )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'elu') )
model.add( Flatten() )
model.add( Dense( 100, activation = 'elu' ) )
model.add( Dropout( 0.50 ) )
model.add( Dense( 50,  activation = 'elu' ) )
model.add( Dropout( 0.25 ) )
model.add( Dense( 10,  activation = 'elu' ) )
model.add( Dropout( 0.1 ) )
model.add( Dense( 1,   activation = 'elu' ) )

# Compile the model and train the neural network
model.compile( loss = 'mse', optimizer = 'adam' )
model.fit_generator( train_generator, samples_per_epoch =
                     len( train_samples ), validation_data = validation_generator,
                     nb_val_samples = len( validation_samples ), nb_epoch = 3 )

# Save the model so we can run the Sim in autonomous mode
model.save( 'model.h5' )
