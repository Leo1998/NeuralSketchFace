import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K
#K.set_image_data_format('channels_first')

WIDTH = 160
HEIGHT = 160




def create_model():
  model = Sequential()

  model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(WIDTH, HEIGHT, 1)))
  model.add(MaxPooling2D((2, 2), padding='same'))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  model.add(MaxPooling2D((2, 2), padding='same'))

  model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
  model.add(MaxPooling2D((2, 2), padding='same'))
  model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
  model.add(MaxPooling2D((2, 2), padding='same'))
  #model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
  #model.add(MaxPooling2D((2, 2), padding='same'))

  ################################################## 

  #model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
  #model.add(UpSampling2D((2, 2)))
  model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
  model.add(UpSampling2D((2, 2)))
  model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
  model.add(UpSampling2D((2, 2)))

  model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  model.add(UpSampling2D((2, 2)))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  model.add(UpSampling2D((2, 2)))

  model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))




  opt=Adam(lr=0.001, decay=0.00001)
  model.compile(loss="mean_squared_error", optimizer=opt)

  model.summary()
  return model


if __name__ == '__main__':
  model = create_model()
 
  print("Loading Data...")
  X = np.load("X.npy")
  Y = np.load("Y.npy")

  X = np.divide(X, 255.0)
  Y = np.divide(Y, 255.0)

  EPOCHS = 50

  model.fit(X, Y, epochs=EPOCHS, batch_size=256)

  model.save("net-{}E.model".format(EPOCHS))

