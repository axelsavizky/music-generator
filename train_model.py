from music21 import *
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from datetime import datetime

from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K
from keras.models import load_model
import multiprocessing
import sys
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from utils import *

path = "/Users/jscherman/GolandProjects/music-generator/gwern/midis/Big_Data_Set"
frequent_notes_threshold = 256

n_of_timesteps = 32
evaluation_percentage = 0.2 # 20% of the data will be used as evaluation

output_dimension = 100
kernel_size = 3
epochs = 50

len_of_predictions = 30

from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
from multiprocess import Pool, set_start_method
import multiprocessing
import time
from functools import partial

from pebble import ProcessPool
from concurrent.futures import TimeoutError

def fibonacci(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return fibonacci(n - 1) + fibonacci(n - 2)

if __name__ == "__main__":
    TRAINING_SET_SIZE = int(sys.argv[1]) ## Cambiar

#     NWORKERS = 1
    NWORKERS = multiprocessing.cpu_count()

    start_time = datetime.now()
    print('starting at: ', start_time)

    files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.mid'))]
    print(f'#files: {str(len(files))} | #training: {TRAINING_SET_SIZE} | #workers: {str(NWORKERS)}')
    start = time.time()
    #files=[i for i in os.listdir(path) if i.endswith(".mid")]

    total = TRAINING_SET_SIZE
    notes_array = []
    read_midi_input = [(file, i, total) for i, file in enumerate(files[:TRAINING_SET_SIZE])]

    read = 0
    notes_array = []
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

    print("read: " + str(read))
    print("#notes_array: " + str(len(notes_array)))
    print('filtering...')
    notes_array = [e for e in notes_array if e.shape[0] > 2]
    print("#filtered_notes_array: " + str(len(notes_array)))

    notes_array = np.array(notes_array, dtype=object)
    print('notes...')
    notes_ = [element for note_ in notes_array for element in note_]
    end = time.time()

    print(f'took {end - start} seconds')
    print(notes_array.shape[0])

    unique_notes = list(set(notes_))
    print(len(unique_notes))

    freq = dict(Counter(notes_))

    no=[count for _,count in freq.items()]

    frequent_notes = [note_ for note_, count in freq.items() if count>=frequent_notes_threshold]
    print(f'keeping {frequent_notes} notes')

    # Get the same dataset only with frequent notes
    new_music=[]

    for notes in notes_array:
        new_music.append([note for note in notes if note in frequent_notes])

    new_music = np.array(new_music, dtype=object)


    inputs = []
    outputs = []

    for notes_ in new_music:
        for i in range(0, len(notes_) - n_of_timesteps, 1):

            inputs.append(notes_[i:i + n_of_timesteps])
            outputs.append(notes_[i + n_of_timesteps])

    inputs=np.array(inputs)
    outputs=np.array(outputs)


    unique_inputs = list(set(inputs.ravel()))
    input_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_inputs))
    import pickle
    filehandler = open(sys.argv[2]+'_dict.pickle','wb')
    pickle.dump(input_note_to_int, filehandler)
    filehandler.close()


    input_seq=[]
    for input_ in inputs:
        input_seq.append([input_note_to_int[note_] for note_ in input_])

    input_seq = np.array(input_seq)


    unique_outputs = list(set(outputs))
    output_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_outputs))
    output_seq = np.array([output_note_to_int[note_] for note_ in outputs])


    input_training, input_validation, output_training, output_validation = train_test_split(input_seq,output_seq,test_size=evaluation_percentage,random_state=0)
    print(f'#input(training/validation): {len(input_training)}/{len(input_validation)}')
    print(f'#output(training/validation): {len(output_training)}/{len(output_validation)}')

    for activation_f in ['relu', 'tanh', 'sigmoid']:

        K.clear_session()
        model = Sequential()

        # activation_f = 'relu'
        # activation_f = 'sigmoid'
        # activation_f = 'tanh'

        # Parameters explanation: https://keras.io/api/layers/core_layers/embedding/
        model.add(Embedding(len(unique_inputs), output_dimension, input_length=n_of_timesteps,trainable=True))

        # Parameters explanation: https://keras.io/api/layers/convolution_layers/convolution1d/
        model.add(Conv1D(n_of_timesteps*2*2,kernel_size, padding='causal',activation=activation_f))
        model.add(Dropout(0.2))
        model.add(MaxPool1D(2))

        model.add(Conv1D(n_of_timesteps*4*2,kernel_size, activation=activation_f,dilation_rate=2,padding='causal'))
        model.add(Dropout(0.2))
        model.add(MaxPool1D(2))

        model.add(Conv1D(n_of_timesteps*8*2,kernel_size, activation=activation_f,dilation_rate=4,padding='causal'))
        model.add(Dropout(0.2))
        model.add(MaxPool1D(2))

        #model.add(Conv1D(256,5,activation=activation_f))
        model.add(GlobalMaxPool1D())

        # Parameters explanation: https://keras.io/api/layers/core_layers/dense/
        # 256 -> 512
        model.add(Dense(512, activation=activation_f))
        model.add(Dense(len(unique_outputs), activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        model_name = f'model_{sys.argv[2]}_{activation_f}'

        epochs = 50

        checkpoint = ModelCheckpoint(model_name, monitor='val_loss', mode='min', save_best_only=True,verbose=1)
        history = model.fit(np.array(input_training),np.array(output_training), batch_size=1024, epochs=epochs, validation_data=(np.array(input_validation),np.array(output_validation)),verbose=1, callbacks=[checkpoint])

        print(f'finish at {datetime.now()} (elapsed {datetime.now() - start_time})')





