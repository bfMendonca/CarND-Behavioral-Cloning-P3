# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Some importante notes should be mentioned here regarding the model.py script. As i discovered during the development proccess, Tensorflow and Keras seems to have a memory leak that makes the memory usage increase during the model training proccess, ` model.fit(..)`. This made the training proccess very changeling as the fit could not be made for more than 5 epochs, otherwise, the RAM would be compromosed and the swap used, what made the training performance to degrade badly.

One workaround solution was to limit the training to be made for only 5 epochs at time `model.fit(..epochs=5` and then saved to be oppened with another script 'retrain.py', which would load the model from the h5 file and continue the training proccess for more 5 epochs. Late in development the checkpoint mechanisms presented at tensorflow were found at the documentation but the scripts were not modified to use this mechanism as workaround for the memory leak. 

Another important aspect is that for test different models and their behaviour at the simulator, different architectures were tested with slightly differences between them to understand the there was any improvement on the vehicle behaviour. This comparison proccess was made so that each architecture was trained with the same dataset and hyperparemeters for the same amount of epochs, to made the comparison more direct. 

Those architectres are NVIDIANetV0, NVIDIANetV1, NVIDIANetV2, NVIDIANetV3, NVIDIANetV4, NVIDIANetV5, NVIDIANetV6, ModNVIDIANetV1, ModNVIDIANetV2, and ModNVIDIANetV3 defined at their respective methods at lines 53, 84, 112, 143, 177, 207, 238, 275 and 310 respectively. 

The models were saved under the scheme 'architecture_name.h5', and loaded with the same procedure at the "retrain.py" script to further training. 

In order to generate and train one specific architecture one could use the following:

```sh
python model.py NVIDIANetV5
python retrain.py NVIDIANetV5
```

Eatch of the following would isse a 5 epochs training to the network. To be noted, the submission network, presented at 'model.h5', was the fifth version for the NVIDIA based architecture, NVIDIANetV5, as it presented the best behaviour at the track testing scneario. This could be verified as the networks had their "name" set during generation, so the submited 'model.h5' will presented the follow at its summary name:

Model: "NVIDIANetV5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda (Lambda)              (None, 160, 320, 3)       0         
_________________________________________________________________

This was the best "catalaling" mechanism that was found during development to keep prototyping and testing sane.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

As mentioned before the architecture used for submission was the "NVIDIANetV5" which was heavily based on the NVIDIA presented during the classes. Other version were testes but presented poor performances compared to that. To better describe the network architecture, the "mode.sumary" is presented as outputed at the terminal:
Model: "NVIDIANetV5"

|Layer (type)               |  Output Shape        |  Param #  |
|---------------------------|:--------------------:|-----------|
|lambda (Lambda)            |  (None, 160, 320, 3) |  0        |
|cropping2d (Cropping2D)    |  (None, 65, 320, 3)  |  0        |
|conv2d (Conv2D)            |  (None, 31, 158, 24) |  1824     |
|conv2d_1 (Conv2D)          |  (None, 14, 77, 36)  |  21636    |
|conv2d_2 (Conv2D)          |  (None, 5, 37, 48)   |  43248    |
|conv2d_3 (Conv2D)          |  (None, 3, 35, 64)   |  27712    |
|conv2d_4 (Conv2D)          |  (None, 1, 33, 64)   |  36928    |
|flatten (Flatten)          |  (None, 2112)        |  0        |
|dense (Dense)              |  (None, 100)         |  211300   |
|dropout (Dropout)          |  (None, 100)         |  0        |
|dense_1 (Dense)            |  (None, 50)          |  5050     |
|dropout_1 (Dropout)        |  (None, 50)          |  0        |
|dense_2 (Dense)            |  (None, 10)          |  510      |
|dense_3 (Dense)            |  (None, 1)           |  11       |
|lambda_1 (Lambda)          |  (None, 1)           |  0        |

Tot|al params: 348,219
Trainable params: 348,219
Non-trainable params: 0


One addition that was made was an output "Lambda Layer" to understand how to make the Network output more generic. For this, using an simplification for the "Ackerman vehicle model", one can verify that the the inverse of the vehicle turning radius, 1/R, tan be described as the tan(alpha)/l, where alpha is the turning radius and the l is the vehicle wheelbase. As the more "generic" output seems to be this inverse of turning Radius, r=1/R, then the steering angle could be isolated as being: alpha = atan(l*r). Assuming that the network output should be this r, the final lambda layer added was proposed in order to conver from r to alpha, and then from radians to deegres to be feed out to the vehicle, as the expeted input for the vehicle was the steering angle in deegres. 

This procedure is presented in all tested network architectures, besides the first one, "NVIDIANetV0". Comparing the output of "NVIDIANetV0" and "NVIDIANetV2", the "NVIDIANetV2" seemed to be more "stable", with more smooth steering angle, so this pattern was assumed to be "positive", and the following architectures inherited that "output lambda layer".

For the submited architectured, NVIDIANetV5, this lambda layer can be seen at line 200. 

#### 2. Attempts to reduce overfitting in the model

To reduce the overfit some testings were made using dropout at the output portion of the network, more specifically, at its dense layers. More specifically The differences between networks versions 4, 5 and 6 were regarding the dropout rate between the dense layers at the end of the Network. 
The model submited, with the best performance observed on the test track, had the dropout configured as follow

    model.add( Flatten( ) )
    model.add( Dense(100,   activation='tanh' ) )
    model.add( Dropout(0.5) )
    model.add( Dense(50,    activation='tanh' ) )
    model.add( Dropout(0.25) )
    model.add( Dense(10,    activation='tanh' ) )   
    model.add( Dense(1,     activation='linear') )

As the number of neurons, and connections decreases, to this one seemed natural to decrease the dropout rate, as with less connections, the effect of dropout would be more severe. 
For the final layer, responsible for estimating the final output, dropout whas not used as there was only 10 neurons from the previous layer connected to this one. This seemed to be reasonable at development and this network showed the best beformance during testing. 

#### 3. Model parameter tuning

The adam optmizer was used, and the training paramters can be seen at line 347, 348 and 349 of "model.py", and "retrain.py", but for documenting reasons those are:

BATCH_SIZE=64
LEARNING_RATE=1e-4
EPOCHS=5

One importante note here is that this EPOCHS value is used for each call of "retrain.py", but the total number of epochs was not 5. In order to make the training more "automatic", an script named "batch_training.sh" was written and countains the following:

#!/bin/bash
python model.py NVIDIANetV0 && for i in {1..9}; do python retrain.py NVIDIANetV3; done
python model.py NVIDIANetV1 && for i in {1..9}; do python retrain.py NVIDIANetV3; done
python model.py NVIDIANetV2 && for i in {1..9}; do python retrain.py NVIDIANetV3; done
python model.py NVIDIANetV3 && for i in {1..9}; do python retrain.py NVIDIANetV3; done
python model.py NVIDIANetV4 && for i in {1..9}; do python retrain.py NVIDIANetV4; done
python model.py NVIDIANetV5 && for i in {1..9}; do python retrain.py NVIDIANetV5; done
python model.py NVIDIANetV6 && for i in {1..9}; do python retrain.py NVIDIANetV6; done
python model.py NVIDIANetV6 && for i in {1..9}; do python retrain.py NVIDIANetV6; done

python model.py ModNVIDIANetV1 && for i in {1..9}; do python retrain.py ModNVIDIANetV1; done
python model.py ModNVIDIANetV2 && for i in {1..9}; do python retrain.py ModNVIDIANetV2; done

As can be seen, each architecture was trained 10 times, 10*5=50 Epochs in total. 

This script was conceibed so that all proposed networks could be training during nights and results evaluated in the morning. 

#### 4. Appropriate training data

For training data the vehicle was driven through the test track for 3 times on each way trying to keep it more centered as possible. After that additionaly for each turn data was collected carefully two times for each way in order to feed the network more examples of how handle turning.

For recovery the vehicle was left, while not recording, on the sides of the road then the recorded started and the vehicle was driven back to the center of the road. This procedure was executed for many portions of the track and in both sides and both ways, in order to have many examples of how to recover if the vehicle "slips" for one side. 

To make the dataset "artifiaclly" bigger i used the "flip" as advised at the course and generated more imagens as than the originally were. This can be seen at the generator_method "load_data", defined at tline 390 with the "augmentation" made on lines 409, 410, 411 and 412. 

Using the left and right cameras and trying to generate more data could not lead to a suscefull model, as the car behaviour becomes erratic while driving, in either case, the lines for using those images were left at the 'model.py' commented out in lines 414 through 428. 

For shuffling and splitting the dataset, as some flags become ununsable on "model.fit()" when using generators, the Pandas library was used in order to read the CSV, split the data into training and validation and then shuffling the results. This procedure is documented at lines 380, 381, 382 and 383. The result of this proccess was two Pandas DataFrame objects, "train" and "valid", passed down to the generator in order to load the images and generate the arrays.

As the training proccess also became "split" due to the need to use the "retrain.py" script to workaround the memory leak, one procces found was to use TensorBoard to log the evolution though the proccess. For that a tensorboard_callback was used, line 352 and the passed down to the 'model.fit(..[tensorboard_callback])' so that the training metrics could be saved and then seem at the Tensorboard interface. The "logdir", needed by the tensorboard, received the "model.name" so that those different architecture could be training and the metrics would note be "mixed up". 