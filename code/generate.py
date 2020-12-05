# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:35:33 2020

@author: Wenjun Ma
"""
import pickle
import numpy
import music21
import tensorflow as tf


def prepare_sequences(notes, pitchnames, n_vocab):
    # note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # sequence_length = 100
    # network_input = []
    # output = []
    # for i in range(0, len(notes) - sequence_length, 1):
    #     seq_in = notes[i:i + sequence_length]
    #     seq_out = notes[i + sequence_length]
    #     network_input.append([note_to_int[char] for char in seq_in])
    #     output.append(note_to_int[seq_out])

    # n_patterns = len(network_input)

    # # Reshape
    # normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalized_input = normalized_input / float(n_vocab)

    # return (network_input, normalized_input)



def create_network(network_input, n_vocab):
    """ Recreate neural network structure. """
    print('Creating network...')

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
    '''file name???''''
    model.load_weights('???.hdf5')
    
    return model
def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from neural net based on input sequence of notes. """
    print('Generating notes...')

    # Pick random sequence from input as starting point
    #length=len(network_input)
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # # Generate n notes
    n = 200
    for note_index in range(n):
    prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)

    # Take most probable prediction, convert to note, append to output
    index = numpy.argmax(prediction)
    #result = int_to_note[index]
    #     prediction_output.append(result)

    #     # Scoot input over by 1 note
    #     pattern.append(index)
    #     pattern = pattern[1:len(pattern)]

    # return prediction_output


def create_midi(prediction_output):
    print('Creating midi...')
    """ Convert prediction output to notes. Create midi file!!!! """
    offset = 0
    output_notes = []
    # # Possible extension: multiple/different instruments!
    stored_instrument = instrument.Piano()

    # # Create Note and Chord objects
    for pattern in prediction_output:
    #     # Pattern is a Chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = stored_instrument
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else: # Pattern is a note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = stored_instrument
            output_notes.append(new_note)

    #     # Increase offset for note
    #     # Possible extension: ~ RHYTHM ~
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output_song.mid')

def main():
    ''''transform input midi file to notes in order to feed it to the input of RNN'''
    notes = []
    midi = music21.converter.parse("../../MusicBot-by-Producer-404/data/cosmo.mid")
    notes_to_parse = midi.parts[0].flat.notes     
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    # Get pitch names
    pitchnames = sorted(set(notes))
    n_vocab = len(set(notes))
    
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)
   
if __name__ == '__main__':
    main()



