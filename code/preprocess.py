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
        notes_to_parse = midi.parts[0].flat.notes     

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
    # prevent overfitting, need to do more research on this
    # with open('data/notes', 'wb') as filepath:
    #     music21.pickle.dump(notes, filepath)

    return notes

def get_data():
    """
    Read the data files and extract notes and chords sequence from the files.
    Create a note dictionary that maps all the unique notes and chords from 
    data as keys to a unique integer value.
    Then vectorize train data based on note dictionary.

    :return: Tuple of train (1-d list with training notes in id form), 
             vocabulary (dict containg note->index mapping)
    """
    test_fraction = 0.1
    notes = get_notes()
    
    # mapping notes to ids
    all_notes = sorted(set(notes))
    notes_dict = {note: i for i, note in enumerate(all_notes)}
    
    # process train data according to notes_dict
    data = []
    for note in notes:
        data.append(notes_dict[note])
    
    test_len = int(len(data) * test_fraction)
    train_data = data[:-test_len]
    test_data = data[-test_len:]
    print(len(data))
    print(len(train_data))
    print(len(test_data))
    
    return train_data, test_data, notes_dict