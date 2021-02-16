import csv
import cv2
import numpy as np
import pandas as pd
import sys
from datetime import datetime

from numpy.random import RandomState

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, MaxPool2D



def DrivingNetV1():
	model = Sequential()
	model.add( Cropping2D( cropping=( (90,20), (0,0) ), input_shape=( 160, 320, 3 ) ) ) 
	model.add( Lambda( lambda x: (x/255.0) - 0.5 ) )
	model.add( Flatten( ) )
	model.add( Dense(1) )

	return model

def NVIDIANetV0( lr=1e-3):
	model = Sequential( name="NVIDIANetV0" )
	
	model.add( Lambda( lambda x: (x/255.0) - 0.5, input_shape=( 160, 320, 3 ) ) )
	model.add( Cropping2D( cropping=( (70,25), (0,0) ) ) )
	
	model.add( Conv2D(  24, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  36, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  48, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )

	model.add( Flatten( ) )
	#model.add( Dense(1164, activation='relu' ) )
	#model.add( Dropout(0.2))

	model.add( Dense(100, 	activation='linear' ) )
	model.add( Dense(50, 	activation='linear' ) )
	model.add( Dense(10, 	activation='linear' ) )	
	model.add( Dense(1, 	activation='linear') )
	
	opt = keras.optimizers.Adam(learning_rate=lr )
	model.compile( loss='mse', optimizer=opt )

	return model

def NVIDIANetV1( lr=1e-3):
	model = Sequential( name="NVIDIANetV1" )
	
	model.add( Lambda( lambda x: (x/255.0) - 0.5, input_shape=( 160, 320, 3 ) ) )
	model.add( Cropping2D( cropping=( (70,25), (0,0) ) ) )
	
	model.add( Conv2D(  24, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  36, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  48, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )

	model.add( Flatten( ) )
	#model.add( Dense(1164, activation='relu' ) )
	#model.add( Dropout(0.2))

	model.add( Dense(100, 	activation='tanh' ) )
	model.add( Dense(50, 	activation='tanh' ) )
	model.add( Dense(10, 	activation='tanh' ) )	
	model.add( Dense(1, 	activation='linear') )

	#Converting curvature to angle, assuming wheelbase of 2 meters, then going from rad to deg
	#The network output was supposed to be 1/R (r), the function then convert it to steering angle (alpha) [deg]
	# alpha = atan(l*r)*57.3. l = wheelbase, supossed to be 2 meters
	model.add( Lambda( lambda x: tf.multiply( tf.atan( tf.multiply( x, 2 ) ), 57.3  )  ) )
	
	opt = keras.optimizers.Adam(learning_rate=lr )
	model.compile( loss='mse', optimizer=opt )

	return model

def NVIDIANetV2( lr=1e-3):
    model = Sequential( name="NVIDIANetV2" )
    
    model.add( Lambda( lambda x: (x/255.0) - 0.5, input_shape=( 160, 320, 3 ) ) )
    model.add( Cropping2D( cropping=( (70,25), (0,0) ) ) )
    
    model.add( Conv2D(  24, 5, 2,   activation='relu', padding='valid' ) )
    model.add( Conv2D(  36, 5, 2,   activation='relu', padding='valid' ) )
    model.add( Conv2D(  48, 5, 2,   activation='relu', padding='valid' ) )
    model.add( Conv2D(  64, 3,      activation='relu', padding='valid' ) )
    model.add( Conv2D(  64, 3,      activation='relu', padding='valid' ) )

    model.add( Flatten( ) )
    model.add( Dense(100,   activation='linear' ) )    
    model.add( Dense(50,    activation='linear' ) )
    model.add( Dense(10,    activation='linear' ) )
    model.add( Dense(1,     activation='linear') )

    #Converting curvature to angle, assuming wheelbase of 2 meters, then going from rad to deg
    #The network output was supposed to be 1/R (r), the function then convert it to steering angle (alpha) [deg]
    # alpha = atan(l*r)*57.3. l = wheelbase, supossed to be 2 meters
    model.add( Lambda( lambda x: tf.multiply( tf.atan( tf.multiply( x, 2 ) ), 57.3  )  ) )
    
    opt = keras.optimizers.Adam(learning_rate=lr )
    model.compile( loss='mse', optimizer=opt )

    return model

def NVIDIANetV3( lr=1e-3):
	model = Sequential( name="NVIDIANetV3" )
	
	model.add( Lambda( lambda x: (x/255.0) - 0.5, input_shape=( 160, 320, 3 ) ) )
	model.add( Cropping2D( cropping=( (70,25), (0,0) ) ) )
	
	model.add( Conv2D(  24, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  36, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  48, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )

	model.add( Flatten( ) )
	model.add( Dense(100, 	activation='tanh' ) )
	model.add( Dropout(0.5) )
	model.add( Dense(50, 	activation='tanh' ) )
	model.add( Dropout(0.5) )
	model.add( Dense(10, 	activation='tanh' ) )	
	model.add( Dropout(0.5) )
	model.add( Dense(1, 	activation='linear') )

	#Converting curvature to angle, assuming wheelbase of 2 meters, then going from rad to deg
	#The network output was supposed to be 1/R (r), the function then convert it to steering angle (alpha) [deg]
	# alpha = atan(l*r)*57.3. l = wheelbase, supossed to be 2 meters
	model.add( Lambda( lambda x: tf.multiply( tf.atan( tf.multiply( x, 2 ) ), 57.3  )  ) )
	
	opt = keras.optimizers.Adam(learning_rate=lr )
	model.compile( loss='mse', optimizer=opt )

	return model

def NVIDIANetV4( lr=1e-3):
	model = Sequential( name="NVIDIANetV4" )
	
	model.add( Lambda( lambda x: (x/255.0) - 0.5, input_shape=( 160, 320, 3 ) ) )
	model.add( Cropping2D( cropping=( (70,25), (0,0) ) ) )
	
	model.add( Conv2D(  24, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  36, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  48, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )

	model.add( Flatten( ) )
	#model.add( Dense(1164, activation='relu' ) )
	#model.add( Dropout(0.2))

	model.add( Dense(100, 	activation='tanh' ) )
	model.add( Dropout(0.5) )
	model.add( Dense(50, 	activation='tanh' ) )
	model.add( Dropout(0.25) )
	model.add( Dense(10, 	activation='tanh' ) )	
	model.add( Dropout(0.125) )
	model.add( Dense(1, 	activation='linear') )

	#Converting curvature to angle, assuming wheelbase of 2 meters, then going from rad to deg
	#The network output was supposed to be 1/R (r), the function then convert it to steering angle (alpha) [deg]
	# alpha = atan(l*r)*57.3. l = wheelbase, supossed to be 2 meters
	model.add( Lambda( lambda x: tf.multiply( tf.atan( tf.multiply( x, 2 ) ), 57.3  )  ) )
	
	opt = keras.optimizers.Adam(learning_rate=lr )
	model.compile( loss='mse', optimizer=opt )

	return model

def NVIDIANetV5( lr=1e-3):
	model = Sequential( name="NVIDIANetV5" )
	
	model.add( Lambda( lambda x: (x/255.0) - 0.5, input_shape=( 160, 320, 3 ) ) )
	model.add( Cropping2D( cropping=( (70,25), (0,0) ) ) )
	
	model.add( Conv2D(  24, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  36, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  48, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )

	model.add( Flatten( ) )
	model.add( Dense(100, 	activation='tanh' ) )
	model.add( Dropout(0.5) )
	model.add( Dense(50, 	activation='tanh' ) )
	model.add( Dropout(0.25) )
	model.add( Dense(10, 	activation='tanh' ) )	
	model.add( Dense(1, 	activation='linear') )

	#Converting curvature to angle, assuming wheelbase of 2 meters, then going from rad to deg
	#The network output was supposed to be 1/R (r), the function then convert it to steering angle (alpha) [deg]
	# alpha = atan(l*r)*57.3. l = wheelbase, supossed to be 2 meters
	model.add( Lambda( lambda x: tf.multiply( tf.atan( tf.multiply( x, 2 ) ), 57.3  )  ) )
	
	opt = keras.optimizers.Adam(learning_rate=lr )
	model.compile( loss='mse', optimizer=opt )

	return model

def NVIDIANetV6( lr=1e-3):
	model = Sequential( name="NVIDIANetV6" )
	
	model.add( Lambda( lambda x: (x/255.0) - 0.5, input_shape=( 160, 320, 3 ) ) )
	model.add( Cropping2D( cropping=( (70,25), (0,0) ) ) )
	
	model.add( Conv2D(  24, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  36, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  48, 5, 2, 	activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )
	model.add( Conv2D(  64, 3, 		activation='relu', padding='valid' ) )

	model.add( Flatten( ) )
	model.add( Dropout(0.5) )
	model.add( Dense(100, 	activation='tanh' ) )
	model.add( Dropout(0.5) )
	model.add( Dense(50, 	activation='tanh' ) )
	model.add( Dropout(0.25) )
	model.add( Dense(10, 	activation='tanh' ) )	
	model.add( Dense(1, 	activation='linear') )

	#Converting curvature to angle, assuming wheelbase of 2 meters, then going from rad to deg
	#The network output was supposed to be 1/R (r), the function then convert it to steering angle (alpha) [deg]
	# alpha = atan(l*r)*57.3. l = wheelbase, supossed to be 2 meters
	model.add( Lambda( lambda x: tf.multiply( tf.atan( tf.multiply( x, 2 ) ), 57.3  )  ) )
	
	opt = keras.optimizers.Adam(learning_rate=lr )
	model.compile( loss='mse', optimizer=opt )

	return model

def ModNVIDIANetV1( lr=1e-3):
	model = Sequential( name = "ModNVIDIANetV1" )
	
	model.add( Lambda( lambda x: (x/255.0) - 0.5, input_shape=( 160, 320, 3 ) ) )
	model.add( Cropping2D( cropping=( (70,25), (0,0) ) ) )
	
	#Keeping padding as "same" and applygin a max
	model.add( Conv2D(  24, 5, 1, activation='relu', padding='same' ) )
	model.add( MaxPool2D( ) )
	model.add( Conv2D(  36, 5, 1, activation='relu', padding='same' ) )
	model.add( MaxPool2D( ) )
	model.add( Conv2D(  48, 5, 1, activation='relu', padding='same' ) )
	model.add( MaxPool2D( ) )
	model.add( Conv2D(  64, 3, 1, activation='relu', padding='same' ) )
	model.add( MaxPool2D( ) )
	model.add( Conv2D(  64, 3, 1, activation='relu', padding='same' ) )
	model.add( MaxPool2D( pool_size=(4, 2) ) )	#forcing to this output to become an "flat"

	model.add( Flatten( ) )
	#model.add( Dense(1164, activation='relu' ) )
	#model.add( Dropout(0.2))
	model.add( Dense(300, 	activation='tanh' ) )
	model.add( Dense(100, 	activation='tanh' ) )
	model.add( Dense(50, 	activation='tanh' ) )
	model.add( Dense(10, 	activation='tanh' ) )
	model.add( Dense(1, 	activation='linear' ) )

	#Converting curvature to angle, assuming wheelbase of 2 meters, then going from rad to deg
	#The network output was supposed to be 1/R (r), the function then convert it to steering angle (alpha) [deg]
	# alpha = atan(l*r)*57.3. l = wheelbase, supossed to be 2 meters
	model.add( Lambda( lambda x: tf.multiply( tf.atan( tf.multiply( x, 2 ) ), 57.3 ) ) )
	
	opt = keras.optimizers.Adam(learning_rate=lr )
	model.compile( loss='mse', optimizer=opt )

	return model

def ModNVIDIANetV2( lr=1e-3):
    model = Sequential( name = "ModNVIDIANetV2" )
    
    model.add( Lambda( lambda x: (x/255.0) - 0.5, input_shape=( 160, 320, 3 ) ) )
    model.add( Cropping2D( cropping=( (70,25), (0,0) ) ) )
    
    #Keeping padding as "same" and applygin a max
    model.add( Conv2D(  24, 5, 1, activation='relu', padding='same' ) )
    model.add( MaxPool2D( ) )
    model.add( Conv2D(  36, 5, 1, activation='relu', padding='same' ) )
    model.add( MaxPool2D( ) )
    model.add( Conv2D(  48, 5, 1, activation='relu', padding='same' ) )
    model.add( MaxPool2D( ) )
    model.add( Conv2D(  64, 3, 1, activation='relu', padding='same' ) )
    model.add( MaxPool2D( ) )
    model.add( Conv2D(  64, 3, 1, activation='relu', padding='same' ) )
    model.add( MaxPool2D( pool_size=(4, 2) ) )  #forcing to this output to become an "flat"

    model.add( Flatten( ) )
    model.add( Dense(300,   activation='linear' ) )
    model.add( Dense(100,   activation='linear' ) )
    model.add( Dense(50,    activation='linear' ) )
    model.add( Dense(10,    activation='linear' ) )    
    model.add( Dense(1,     activation='linear' ) )

    #Converting curvature to angle, assuming wheelbase of 2 meters, then going from rad to deg
    #The network output was supposed to be 1/R (r), the function then convert it to steering angle (alpha) [deg]
    # alpha = atan(l*r)*57.3. l = wheelbase, supossed to be 2 meters
    model.add( Lambda( lambda x: tf.multiply( tf.atan( tf.multiply( x, 2 ) ), 57.3 ) ) )
    
    opt = keras.optimizers.Adam(learning_rate=lr )
    model.compile( loss='mse', optimizer=opt )

    return model

def ModNVIDIANetV3( lr=1e-3):
	model = Sequential( name = "ModNVIDIANetV3" )

	model.add( Lambda( lambda x: (x/255.0) - 0.5, input_shape=( 160, 320, 3 ) ) )
	model.add( Cropping2D( cropping=( (70,25), (0,0) ) ) )

	#Keeping padding as "same" and applygin a max
	model.add( Conv2D(  24, 5, 1, activation='relu', padding='same' ) )
	model.add( MaxPool2D( ) )
	model.add( Conv2D(  36, 5, 1, activation='relu', padding='same' ) )
	model.add( MaxPool2D( ) )
	model.add( Conv2D(  48, 5, 1, activation='relu', padding='same' ) )
	model.add( MaxPool2D( ) )
	model.add( Conv2D(  64, 3, 1, activation='relu', padding='same' ) )
	model.add( MaxPool2D( ) )
	model.add( Conv2D(  64, 3, 1, activation='relu', padding='same' ) )
	model.add( MaxPool2D( pool_size=(4, 2) ) )  #forcing to this output to become an "flat"

	model.add( Flatten( ) )
	model.add( Dense(100, 	activation='tanh' ) )
	model.add( Dropout(0.5) )
	model.add( Dense(50, 	activation='tanh' ) )
	model.add( Dropout(0.25) )
	model.add( Dense(10, 	activation='tanh' ) )	
	model.add( Dense(1, 	activation='linear') )

	#Converting curvature to angle, assuming wheelbase of 2 meters, then going from rad to deg
	#The network output was supposed to be 1/R (r), the function then convert it to steering angle (alpha) [deg]
	# alpha = atan(l*r)*57.3. l = wheelbase, supossed to be 2 meters
	model.add( Lambda( lambda x: tf.multiply( tf.atan( tf.multiply( x, 2 ) ), 57.3 ) ) )

	opt = keras.optimizers.Adam(learning_rate=lr )
	model.compile( loss='mse', optimizer=opt )

	return model

#Hyper parameters
BATCH_SIZE=64
LEARNING_RATE=1e-4
EPOCHS=5

model_name = sys.argv[1]

model = Sequential()

if( model_name == 'NVIDIANetV0'):
	model = NVIDIANetV0( LEARNING_RATE )
elif( model_name == 'NVIDIANetV1'):
    model = NVIDIANetV1( LEARNING_RATE )
elif( model_name == 'NVIDIANetV2' ):
    model = NVIDIANetV2( LEARNING_RATE )
elif( model_name == 'NVIDIANetV3' ):
    model = NVIDIANetV3( LEARNING_RATE )    
elif( model_name == 'NVIDIANetV4' ):
    model = NVIDIANetV4( LEARNING_RATE )    
elif( model_name == 'NVIDIANetV5' ):
    model = NVIDIANetV5( LEARNING_RATE )    
elif( model_name == 'NVIDIANetV6' ):
    model = NVIDIANetV6( LEARNING_RATE )    
elif( model_name == 'ModNVIDIANetV1' ):
    model = ModNVIDIANetV1( LEARNING_RATE )
elif( model_name == 'ModNVIDIANetV2' ):
    model = ModNVIDIANetV2( LEARNING_RATE )
elif( model_name == 'ModNVIDIANetV3' ):
    model = ModNVIDIANetV3( LEARNING_RATE )

else:
    raise Exception('Invalid model name')

#Load data. Split data into train and validation
df = pd.read_csv('data/driving_log.csv', names=['center', 'left', 'right', 'measurement', '1', '2', '3'])
rng = RandomState()
train = df.sample( frac=0.7, random_state=rng )
valid = df.loc[~df.index.isin(train.index) ]


NUM_TRAIN_IMAGES = train.shape[0]
NUM_TEST_IMAGES = valid.shape[0]

#Deffining the generator
def load_data( df, batch_size, augument=False ):
	
	i = 0

	while True:
		
		images = []
		measurements = []

		while len(images) < batch_size:
			
			image_path = df.iloc[i,:]['center'].split('/')[-1]
			current_path = './data/IMG/' + image_path
			measurement = float( df.iloc[i,:]['measurement'] )
			image = cv2.imread( current_path )

			measurements.append( measurement )
			images.append( image )

			if( augument ):
				flipped_image = cv2.flip( image, 1 )
				images.append( flipped_image )
				measurements.append( -1.0*measurement )

			# image_path = df.iloc[i,:]['left'].split('/')[-1]
			# current_path = './data/IMG/' + image_path
			# measurement = float( +0.9 ) 
			# image = cv2.imread( current_path )

			# measurements.append( measurement )
			# images.append( image )

			# image_path = df.iloc[i,:]['right'].split('/')[-1]
			# current_path = './data/IMG/' + image_path
			# measurement = float( -0.9 ) 
			# image = cv2.imread( current_path )

			# measurements.append( measurement )
			# images.append( image )

			i += 1

			if( i == df.shape[0] ):
				i =0


		yield ( np.array( images ), np.array( measurements ) )

#Define the generators
trainGen = load_data( train, BATCH_SIZE, True)
validGen =  load_data( valid, BATCH_SIZE )

NUM_TRAIN_IMAGES = 2*NUM_TRAIN_IMAGES
NUM_TEST_IMAGES = NUM_TEST_IMAGES

print(model.summary())

#Using tensorboard
logdir = "logs/scalars/" + model.name


#defiining tensorboard callback
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model.fit( 
	x=trainGen, 
	steps_per_epoch=NUM_TRAIN_IMAGES//BATCH_SIZE, 
	verbose=1,
	validation_data=validGen,
	validation_steps=NUM_TEST_IMAGES//BATCH_SIZE, epochs=EPOCHS,
	callbacks=[tensorboard_callback] )

model.save( model.name + '.h5')
