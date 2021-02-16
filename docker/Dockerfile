FROM tensorflow/tensorflow:2.4.1-gpu
#FROM tensorflow/tensorflow:1.15.5-gpu-py3
#FROM nvcr.io/nvidia/tensorflow:20.01-tf1-py3
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip3 install setuptools --upgrade
RUN pip3 install opencv-python sklearn eventlet Pillow flask keras python-socketio==4.2.1 pandas 	moviepy

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6 netbase -y

ENV TF_FORCE_GPU_ALLOW_GROWTH true

RUN mkdir -p /home/car_nd && cd /home/car_nd

WORKDIR /home/car_nd
