#!/bin/bash

# create necessary directories
cd ..
mkdir data
cd data
mkdir song-lyrics-dataset
cd song-lyrics-dataset

# download dataset zip files
curl -L -o ./song-lyrics-dataset.zip https://www.kaggle.com/api/v1/datasets/download/deepshah16/song-lyrics-dataset

# unzip dataset
unzip song-lyrics-dataset.zip

# rename 'json files' directory to json
mv 'json files' json
