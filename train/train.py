#This file is for training the model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#Globals
BATCH_SIZE = 64
INPUT_SHAPE = (48,48,1)



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'FER/images/train',
        target_size=(48, 48),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        'FER/images/validation',
        target_size=(48, 48),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical')

def create_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

model = create_model()

def train(model,train_generator,test_generator):
    optimizer = Adam(lr=0.0001,decay=1e-6)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(train_generator,
                                  epochs=10,
                                  steps_per_epoch=28709 // BATCH_SIZE,
                                  validation_steps=7178 // BATCH_SIZE,
                                  validation_data=test_generator)

    return history

hist = train(model,train_generator,test_generator)

def plot_hist(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['train loss','test loss'])
    plt.show()

plot_hist(hist)
