from tensorflow.keras.models import load_model
import pickle
from glob import glob
from multiprocess import Pool
import time
import os
import os.path
import sys
from music21 import *
import numpy as np 

model_path = sys.argv[1]
dict_path = sys.argv[2]
TEST_SET_SIZE = sys.argv[3]




def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())

def get_predictions(model, song):
    N = len(song)
    predictions = []
    for i in range(1, N ):
        # Get the first i notes
        
        if i < 32:
            partial_song = song[:i]
        else:
            partial_song = song[i-32:i]
        # Pad it with zeroes to get 32 notes
        partial_song_with_padding = np.pad(partial_song, (max(32 - len(partial_song), 0), 0), 'constant', constant_values=0)
        
        # Evaluate it
        prob = model.predict(partial_song_with_padding.reshape(1, 32))

        predictions.append(prob[0][song[i]])
    return predictions

def stable_perplexity(predictions):
    N = len(predictions)
    log_perplexity = 0
    for prob in predictions:
        log_perplexity += np.log(prob)
    return np.exp(-log_perplexity/float(N))

def default_perplexity(predictions):
    N = len(predictions)
    perplexity = 1
    for prob in predictions:
        perplexity*=prob
    return (1/perplexity)**(1/float(N))

def calculate_perplexity(model, song):
    return default_perplexity(get_predictions(model, song))

def calculate_perplexity_stable(model, song):    
    return stable_perplexity(get_predictions(model, song))


def read_midi(file):
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
    #Looping over all the instruments
    for part in s2.parts:
    
        #select elements of only piano
        if 'Piano' in str(part): 
        
            notes_to_parse = part.recurse() 
      
            #finding whether a particular element is note or a chord
            for element in notes_to_parse:
                
                #note
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                
                #chord
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return np.array(notes)

def feedforward(model_path, model_dict_path, test_path):
    model = load_model(model_path)
    note_to_num_dict = pickle.load( open( model_dict_path, "rb" ) )
    
    # make test dataset
    TEST_SET_SIZE = int(sys.argv[3])
    NTHREADS = 16

    files = [y for x in os.walk(test_path) for y in glob(os.path.join(x[0], '*.mid'))]
    print(len(files))
    start = time.time()
    with Pool(NTHREADS) as p:
        notes_array = p.map(read_midi, files[32000:32000+TEST_SET_SIZE])

    print('filtering...')
    notes_array = [e for e in notes_array if e != np.array([])]
    assert len(notes_array) > 0, "ERROR : songs array is empty!"
    notes_array = np.array(notes_array, dtype=object)
    print('notes...')
    
    end = time.time()

    print(f'took {end - start} seconds')
    input_seq=[]
    for input_ in notes_array:
        input_seq.append([note_to_num_dict[note_] for note_ in input_ if note_ in note_to_num_dict.keys()])

    input_seq = np.array(input_seq)
    
    start = time.time()
    perplexities =  [calculate_perplexity_stable(model, song) for song in input_seq]
    end = time.time()
    print(f'took {end - start} seconds')
    print(perplexities)
    return perplexities
    
result = feedforward(sys.argv[1], sys.argv[2], 'gwern/midis/')

print('--- \n perplexity is: ')
print(str(geo_mean_overflow(result)))
print('---')
