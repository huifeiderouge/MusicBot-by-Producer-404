import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers
from preprocess import get_data

def build_model(vocab_size, embedding_size):
    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_size))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.BatchNorm())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNorm())
    model.add(layers.Dropout(.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNorm())
    model.add(layers.Dropout(.3))
    model.add(layers.Dense(vocab_size))
    model.add(layers.Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam')
    
    return model
