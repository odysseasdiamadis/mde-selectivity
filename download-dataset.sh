#!/bin/sh

DATA_DIR='./data'

mkdir -p $DATA_DIR

curl -L -o $DATA_DIR/nyu-depth-v2.zip \
  https://www.kaggle.com/api/v1/datasets/download/soumikrakshit/nyu-depth-v2

unzip $DATA_DIR/nyu-depth-v2.zip -d $DATA_DIR