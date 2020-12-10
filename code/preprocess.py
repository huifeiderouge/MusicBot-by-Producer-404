import tensorflow as tf
import numpy as np
from music21 import *
import glob

def get_notes():
    """ 
    Get all the notes and chords from the midi files in the ./data directory
    
    :return notes: 1-d list with all notes and chords represented as strings
    """
    notes = []

    for file in glob.glob('../../MusicBot-by-Producer-404/data/*.mid'):
        midi = converter.parse(file)
        # midi = music21.converter.parse("../../MusicBot-by-Producer-404/data/cosmo.mid")
    
        print("Parsing %s" % file)
        
        # Only parse notes and chords in first track, which usually contains the main melody
        notes_to_parse = midi.parts[0].flat.notesAndRests     

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
            elif isinstance(element, note.Rest):
                notes.append('Rest')
        
        notes.append('STOP')
                
    # prevent overfitting, need to do more research on this
    # with open('data/notes', 'wb') as filepath:
    #     music21.pickle.dump(notes, filepath)

    return notes


# split a univariate sequence into samples
def split_data(data, window_size, stop_id):
    inputs, labels = [], []
    i = 0
    while i < (len(data) - window_size):
        # find the end of this pattern
        end_idx = i + window_size
        # gather input and labels parts of the pattern
        inputs.append(data[i:end_idx])
        labels.append(data[end_idx])
        
        if data[end_idx] is stop_id:
            i = end_idx + 1
        else:
            i += 1
        
    inputs = np.array(inputs)
    inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1))
    return inputs, np.array(labels)


def get_data():
    """
    Read the data files and extract notes and chords sequence from the files.
    Create a note dictionary that maps all the unique notes and chords from 
    data as keys to a unique integer value.
    Then vectorize train data based on note dictionary.

    :return: Tuple of train (1-d list with training notes in id form), 
             vocabulary (dict containg note->index mapping)
    """
    test_fraction = 0.2
    window_size = 10
    notes = get_notes()
    
    # mapping notes to ids
    all_notes = sorted(set(notes))
    notes_dict = {note: i for i, note in enumerate(all_notes)}
    
    # process train data according to notes_dict
    data = []
    for note in notes:
        data.append(notes_dict[note])
        
    inputs, labels = split_data(data, window_size, notes_dict['STOP'])
    
    random_ind = tf.random.shuffle([i for i in range(labels.shape[0])])
    inputs = tf.gather(inputs, random_ind)
    labels = tf.gather(labels, random_ind)
    
    test_len = int(len(inputs) * test_fraction)
    train_inputs = inputs[:-test_len]
    train_labels = labels[:-test_len]
    test_inputs = inputs[-test_len:]
    test_labels = labels[-test_len:]
    
    # normalize inputs
    # train_inputs = train_inputs/len(notes_dict)
    # test_inputs = test_inputs/len(notes_dict)
    
    # print(train_inputs.shape)
    # print(train_labels.shape)
    return train_inputs, train_labels, test_inputs, test_labels, notes_dict