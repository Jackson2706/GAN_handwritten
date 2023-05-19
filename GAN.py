from keras.models import Sequential
from keras.layers import *
from matplotlib import pyplot as plt
import numpy as np
import cv2
from numpy.random import randn, randint
import os
from keras.datasets.mnist import load_data as origin_load_data
from keras.optimizers import Adam
from keras.models import Sequential, Model
import tqdm
from sklearn.model_selection import train_test_split   

def adam_optimizer():
    return Adam(learning_rate=1e-4, beta_1=0.3)

def create_generator(latent_dim = 100):
    model = Sequential()
    model.add(Dense(7*7*512, input_dim=latent_dim, activation='relu'))

    new_shape = (7,7,512)
    model.add(Reshape(new_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(filters=256, kernel_size=(3,3), padding='same', strides=2, bias=False))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(filters=128, kernel_size=(4,4), padding='same', strides=2, bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, kernel_size=(5,5), strides=1, padding='same', activation='tanh', bias=False))

    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer())

    return model