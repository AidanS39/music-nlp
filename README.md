# Natural Language Processing applied to Music

Using Natural Language Processing concepts with Python to analyze music lyrics.

## Summary

This project uses NLP concepts such as Naive Bayes, Text Classification, Supervised Machine Learning, and Laplace Smoothing to create language models based off the lyrics of many songs. The project is currently coded entirely using raw Python, meaning only basic Python libraries such as the `math` and `random` libraries are used. In the future, I plan to build on this project, using different (and more sophisticated) concepts to analyze music and evaluate performance between different models.

## Usage
### Requirements
Although this project should be pretty flexible, it is not guaranteed to work without fulfilling these requirement.
1. `Python 3.10.11`: This project was developed and tested using Python 3.10.11.
2. `Linux-based shell`: Since the program prints output to the terminal, a shell that is able to run Python is needed. This project was developed on a Linux terminal.
3. `Significant computing power and memory`: ML tasks are known for using lots of memory and computing power. If on a home machine, the program might execute very slowly or run out of memory.
### Deployment
Follow these steps to execute the project:
1. Download the *Song Lyrics Dataset*. Unfortunately the datasets used for this program are too large for GitHub to store, so they must be downloaded locally. This can be done using the `data_setup.sh` script in the **src** directory (recommended), or by going to https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset and downloading the zip file there.

If you directly downloaded the zip file from the website, you will have to create the correct path to the target data. First create a directory called **data** from the root directory of this project. Then, create another directory inside the **data** folder, which will be named **song-lyrics-dataset**. Then, extract the contents of the zip file into the **song-lyrics-dataset** folder. You should now have the correct path to the target data, which from the root directory should be **data/song-lyrics-dataset/csv**

2. From within the **src** directory, run `python3 music.py`. This should start the main program which will give you a menu to choose from after a few seconds.

3. Choose an option listed in the menu by typing the option's number + **ENTER**, or just press **ENTER** to exit the program.
