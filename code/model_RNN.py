import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__ (self, vocab_size, window_size, embedding_size, batch_size, rnn_size, hidden_layer):
        """
        The Model class predicts the next notes in a sequence.

        :param vocab_size: The number of unique notes in the data
        """
        
        super(Model, self).__init__()
        
        # Initialize vocab_size, emnbedding_size
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.hidden_layer = hidden_layer
        
        # Initialize embeddings and forward pass weights (weights, biases)
        self.E = tf.Variable(tf.random.normal(shape=[vocab_size, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.LSTMLayer = tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=True)
        self.denseLayer1 = tf.keras.layers.Dense(hidden_layer, activation='relu')
        self.denseLayer2 = tf.keras.layers.Dense(vocab_size, activation='softmax')
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def call (self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: note ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state using LSTM and only the probabilities as a
        tensor and a final_state as a tensor when using GRU
        """
        embedding = tf.nn.embedding_lookup(self.E, inputs)
        whole_seq_output, final_memory_state, final_carry_state = self.LSTMLayer(embedding, None)
        outputDense1 = self.denseLayer1(whole_seq_output)
        probabilities = self.denseLayer2(outputDense1)
        
        return probabilities
    
    def loss (self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        l = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False, axis=-1)
        
        return tf.reduce_mean(l)
    
    def train (model, train_inputs, train_labels):
        """
        Runs through one epoch - all training examples.
        
        :param model: the initilized model to use for forward and backward pass
        :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
        :param train_labels: train labels (all labels for training) of shape (num_labels,)
        :return: None
        """
        # TODO: Fill in
        print("Train starts")
        batchSize = model.batch_size
        windowSize = model.window_size
        optimizer = model.optimizer
        
        length = len(train_inputs)
        times = int(length / windowSize)
        
        trainInputSliced = train_inputs[0: times * windowSize]
        trainLabelsSliced = train_labels[0: times * windowSize]
        
        trainInputReshaped = trainInputSliced.reshape(times, windowSize)
        trainLabelsReshaped = trainLabelsSliced.reshape(times, windowSize)
        
        lossArray = []
        EW = []
        LSTMLayerW = []
        dense1W = []
        dense2W = []
        totalLoss = 0
        for i in range(0, len(trainInputReshaped), batchSize):
            batchedInput = trainInputReshaped[i: i + batchSize, :]
            batchedLabel = trainLabelsReshaped[i: i + batchSize, :]
            with tf.GradientTape() as tape:
                pred = model.call(batchedInput, None)
                loss = model.loss(pred, batchedLabel)
            totalLoss = totalLoss + loss
            lossArray.append(totalLoss / (i + 1))
            EW.append(model.E.get_weights())
            LSTMLayerW.append(model.LSTMLayer.get_weights())
            dense1W.append(model.denseLayer1.get_weights())
            dense2W.append(model.denseLayer2.get_weights())
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print("Train ends")
        
        return lossArray, EW, LSTMLayerW, dense1W, dense2W
    
    def generate_node (word1, length, vocab, model, sample_n=10):
        
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        previous_state = None
        
        first_string = word1
        first_word_index = vocab[word1]
        next_input = [[first_word_index]]
        text = [first_string]
        
        for i in range(length):
            logits, previous_state = model.call(next_input, previous_state)
            logits = np.array(logits[0, 0, :])
            top_n = np.argsort(logits)[-sample_n:]
            n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n, p=n_logits)
            
            text.append(reverse_vocab[out_index])
            next_input = [[out_index]]
        
        return text
    
    def main ():
        # TO-DO: Pre-process and vectorize the data
        # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
        # from train_x and test_x. You also need to drop the first element from train_y and test_y.
        # If you don't do this, you will see impossibly small perplexities.
        
        # TO-DO:  Separate your train and test data into inputs and labels
        print("Main starts")
        trainOut, notes_dict = get_data()
        
        trainInputs = np.array(trainOut[:-1])
        trainLabels = np.array(trainOut[1:])
        
        # TODO: initialize model and tensorflow variables
        # vocab_size, window_size, embedding_size, batch_size, rnn_size, hidden_layer
        window_size = ?
        embedding_size = ?
        batch_size = ?
        rnn_size = ?
        hidden_layer = ?
        model = Model(len(notes_dict), window_size, embedding_size, batch_size, rnn_size, hidden_layer)
        
        # TODO: Set-up the training step
        lossArray, EW, LSTMLayerW, dense1W, dense2W = train(model, trainInputs, trainLabels)
        indexOfMin = np.argmin(lossArray)
        model.E.set_weights(EW[indexOfMin])
        model.LSTMLayer.set_weights(LSTMLayerW[indexOfMin])
        model.denseLayer1.set_weights(dense1W[indexOfMin])
        model.denseLayer2.set_weights(dense2W[indexOfMin])
        
        firstNode = ?
        length = ?
        text = generate_node(word1, length, notes_dict, model, sample_n=10)
    
    if __name__ == '__main__':
        main()
