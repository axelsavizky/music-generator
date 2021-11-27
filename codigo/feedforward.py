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
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import multiprocessing

from utils import *

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

def load(test_path, test_set_size):
    start = time.time()
    files = [y for x in os.walk(test_path) for y in glob(os.path.join(x[0], '*.mid'))]
    print(len(files))
    start = time.time()

    total = test_set_size
    notes_array = []
    read_midi_input = [(file, i, total) for i, file in enumerate(files[32000:32000+test_set_size])]
    notes_array = []
    read = 0
    NWORKERS = multiprocessing.cpu_count()
    with ProcessPool() as pool:
        future = pool.map(read_midi_wrapper, read_midi_input, timeout=10, max_workers=NWORKERS)
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
                notes_array.append(result)
                read += 1
            except StopIteration:
                print ("stop iteration!")
                break
            except TimeoutError as error:
                print ("couldn't read midi")

    print('filtering...')
    notes_array = [e for e in notes_array if e != np.array([])]
    assert len(notes_array) > 0, "ERROR : songs array is empty!"
    notes_array = np.array(notes_array, dtype=object)
    end = time.time()
    print(f'took {end - start} seconds')
    return notes_array

def feedforward(model_path, model_dict_path, notes_array):
    model = load_model(model_path)
    note_to_num_dict = pickle.load( open( model_dict_path, "rb" ) )
    print(f'model_path: {model_path}')

    print('notes...')

    input_seq=[]
    for input_ in notes_array:
        input = [note_to_num_dict[note_] for note_ in input_ if note_ in note_to_num_dict.keys()]
        if len(input) > 0:
            input_seq.append(input)

    input_seq = np.array(input_seq)
    assert len(input_seq) > 0

    start = time.time()
    lengths = [len(song) for song in input_seq]
    perplexities =  [calculate_perplexity_stable(model, song) for song in input_seq if len(song) > 2]
    end = time.time()
    print(f'took {end - start} seconds')
    print(perplexities)
    return perplexities

if __name__ == "__main__":
    model_path = sys.argv[1]
    dict_path = sys.argv[2]
    TEST_SET_SIZE = sys.argv[3]

    notes_array = load('gwern/midis/', int(sys.argv[3]))
    result = feedforward(sys.argv[1], sys.argv[2], notes_array)

    print('--- \n perplexity is: ')
    print(str(geo_mean_overflow(result)))
    print('---')
