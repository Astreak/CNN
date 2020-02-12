import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf



data_train = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

data_test= tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

X_train=data_train.flow_from_directory(
        'data/training_set',
        target_size=(64,64),
        batch_size=64,
        class_mode='binary')

x_test=data_test.flow_from_directory(
        'data/test_set',
        target_size=(64,64),
        batch_size=64,
        class_mode='binary')


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(32,(3,3),activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


model.fit_generator(
        X_train,
        #steps_per_epoch=2000,
        epochs=13,
        validation_data=x_test
        #validation_steps=800
        )