from tensorflow.keras.models import load_model
import pickle
from glob import glob
from multiprocess import Pool
import time
import os
import os.path
import sys

model_path = 'model_test_1024.h5'
dict_path = 'model_test_1024.h5_dict.pickle'
def read_midi(file):
    
    print("Reading: " + file)
    
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
    #files=[i for i in os.listdir(path) if i.endswith(".mid")]
    with Pool(NTHREADS) as p:
        notes_array = p.map(read_midi, files[10000:TEST_SET_SIZE])

    print('filtering...')
    notes_array = [e for e in notes_array if e != np.array([])]

    notes_array = np.array(notes_array, dtype=object)
    print('notes...')
    
    end = time.time()

    print(f'took {end - start} seconds')
    input_seq=[]
    for input_ in notes_array:
        input_seq.append([note_to_num_dict[note_] for note_ in input_ if note_ in note_to_num_dict.keys()])

    input_seq = np.array(input_seq)
    
    return np.asarray([calculate_perplexity_stable(model, song) for song in input_seq]).mean()
    
feedforward(sys.argv[1], sys.argv[2], 'gwern/midis/')
