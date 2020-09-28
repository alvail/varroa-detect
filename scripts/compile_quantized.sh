#!/bin/bash
# Execute from project directory to compile TFLite models for Edge TPU

dirs=($(find "models/" -mindepth 1 -maxdepth 1 -type d))
for dir in "${dirs[@]}"; do
  ( cd "$dir" && edgetpu_compiler model.tflite )
done
