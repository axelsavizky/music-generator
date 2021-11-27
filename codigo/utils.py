"""
utils.py
~~~~~~~~~~

A module for common functions shared all project.
"""
from music21 import *
import numpy as np

def read_midi_wrapper(args):
    return read_midi(*args)

def read_midi(file, current = 0, total = 0):
    print(f'reading: {str(current)}/{str(total)}', flush=True)
    notes=[]
    notes_to_parse = None

    #parsing a midi file
    try:
        midi = converter.parse(file)
    except:
        return np.array([])

    #grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)
    if not s2:
        return np.array([])

    monophonics = [
        'Flute',
        'Baritone Saxophone',
        'Trombone',
        'Tenor Saxophone',
        'Clarinet',
        'Alto Saxophone',
        'Bass',
        'Voice',
        'Acoustic Bass',
        'Trumpet',
        'Oboe']

    #Looping over all the instruments
    for part in s2.parts:

        #select elements of only monophonic instrument
        if any(x in str(part) for x in monophonics):

            # Transpose to C major / A minor
            try:
                key = part.analyze('key')
                tonic = key.getTonic()
            except:
                return np.array([])

            if key.mode == 'major':
                part.transpose(tonic.pitchClass * -1, inPlace=True)
            else:
                part.transpose((tonic.pitchClass * -1) - 3, inPlace=True)

            notes_to_parse = part.recurse()

            #finding whether a particular element is note or a chord
            for element in notes_to_parse[:1024]:
#                 print(f'notes to parse: {len(list(notes_to_parse))}')
                #note
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))

                #chord
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    print(f'finish reading: {str(current)}/{str(total)}', flush=True)
    return np.array(notes)