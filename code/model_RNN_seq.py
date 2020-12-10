import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from preprocess import get_data
from music21 import instrument, note, stream, chord

def build_model(vocab_size, window_size, num_features):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(256, return_sequences=True, input_shape=(window_size, num_features)))
    model.add(layers.LSTM(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))#, kernel_initializer='truncated_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.3))
    model.add(layers.Dense(256, activation='relu'))#, kernel_initializer='truncated_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.3))
    model.add(layers.Dense(256, activation='relu'))#, kernel_initializer='truncated_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.3))
    model.add(layers.Dense(vocab_size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.002))
    
    return model


def train(model, inputs, lables, n_epoch):
    print("Training...")
    history = tf.keras.callbacks.History()
    model.fit(inputs, lables, epochs=n_epoch, batch_size=128, callbacks=[history]) # , callbacks=callbacks_list)
    
    print(history.History)
    print("Training finished")
    
    # save weights for generate midi
    model.save("my_weights.h5")


def test(model, inputs, lables):
    print("\nTesting...")
    test_loss = model.evaluate(inputs, lables)
    print("Testing finished")
    return test_loss
    

def main():
    # read in data
    train_inputs, train_labels, test_inputs, test_labels, notes_dict = get_data()
    # build model
    model = build_model(len(notes_dict), 10, 1)
    model.summary()
    # training
    train(model, train_inputs, train_labels, 50)
    # testing
    test_loss = test(model, test_inputs, test_labels)
    print('Test perplexity:{}'.format(test_loss))
    
    
if __name__ == '__main__':
    main()