#!/bin/sh
# For compiling a fully 8-Bit quantized TFLite model for Edge TPU
# Usage: edgetpu_compiler model.tflite
# Refer to https://coral.ai/docs/edgetpu/compiler/#usage for details

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add 
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
apt-get update
apt-get install edgetpu-compiler
