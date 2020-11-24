import tensorflow as tf
import numpy as np
from music21 import *
import glob

def get_notes(file_name):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob(file_name):
        midi = converter.parse(file)
        # midi = music21.converter.parse("../../MusicBot-by-Producer-404/midi_songs/cosmo.mid")
    
        print("Parsing %s" % file)

        notes_to_parse = midi.parts[0].flat.notes     

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
    # 
    # with open('data/notes', 'wb') as filepath:
    #     music21.pickle.dump(notes, filepath)

    return notes