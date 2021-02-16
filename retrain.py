import csv
import cv2
import numpy as np
import pandas as pd
import sys
from datetime import datetime

from numpy.random import RandomState

import keras
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, MaxPool2D


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

#Hyper parameters
BATCH_SIZE=64
LEARNING_RATE=1e-4
EPOCHS=5

#Define the generators
trainGen = load_data( train, BATCH_SIZE, True)
validGen =  load_data( valid, BATCH_SIZE )

NUM_TRAIN_IMAGES = 2*NUM_TRAIN_IMAGES
NUM_TEST_IMAGES = NUM_TEST_IMAGES

#checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=5 )

model_filename = sys.argv[1]
print( model_filename )

#Loading model
model = load_model( model_filename + '.h5' )
print(model.summary())

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
    #,callbacks=[checkpoint] )

model.save( model_filename + '.h5' )