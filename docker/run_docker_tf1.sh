#!/bin/bash
#docker run -u $(id -u):$(id -g) --gpus all -it --rm -v /home/bmendonca/workspace/cnn:/home/cnn tensorflow/tensorflow:1.15.4-py3
#docker run -u $(id -u):$(id -g) --gpus all --net host -it --rm -v --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 /home/bmendonca/workspace/cnn:/home/cnn tensorflow/custom-tf1
docker run -u $(id -u):$(id -g) --gpus all --net host -it --rm -v $(pwd):/home/car_nd tensorflow/behavioral-tf1:latest
