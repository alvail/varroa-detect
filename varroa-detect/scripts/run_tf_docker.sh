#!/bin/sh
# Downloads and executes Docker environment for training models
# Needs to have nvidia-container-toolkit installed

DETECT_DIR=${PWD/..}
docker pull tensorflow/tensorflow:2.3.0-gpu
docker run --gpus all -it --rm --mount type=bind,src=${DETECT_DIR},dst=/varroa-detect/ tensorflow/tensorflow:2.3.0-gpu
