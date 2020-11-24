import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next notes in a sequence.

        :param vocab_size: The number of unique notes in the data
        """

        super(Model, self).__init__()

        # Initialize vocab_size, emnbedding_size
        self.vocab_size = vocab_size
        self.window_size = ?
        self.embedding_size = ?
        self.batch_size = ?
        self.rnn_size = ?
        self.hidden_layer = ?

        # Initialize embeddings and forward pass weights (weights, biases)
        
        

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: note ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state using LSTM and only the probabilities as a
        tensor and a final_state as a tensor when using GRU
        """
        
        return


    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        
        return
