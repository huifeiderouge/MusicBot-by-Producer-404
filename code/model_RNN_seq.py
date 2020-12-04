import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from preprocess import get_data

def build_model(vocab_size, embedding_size):
    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_size))
    model.add(layers.LSTM(512, return_sequences=True))
    model.add(layers.LSTM(512, return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.3))
    model.add(layers.Dense(vocab_size))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.005))
    
    return model

def train(model, inputs, lables):
    print("Training...")
    filepath = "../../MusicBot-by-Producer-404/code/weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(inputs, lables, epochs=50, batch_size=128, callbacks=callbacks_list)
    print("Training finished")

def test(model, inputs, lables):
    print("\nTesting...")
    model.evaluate(inputs, lables)
    print("Testing finished")

    
def main():
    train_data, test_data, notes_dict = get_data()
    model = build_model(len(notes_dict), 128)
    train(model, train_data[:-1], train_data[1:])
    test(model, test_data[:-1], test_data[1:])
    
if __name__ == '__main__':
    main()