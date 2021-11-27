from tensorflow.keras.models import load_model
import pickle
import time
import os
import os.path
import sys
from music21 import *
import numpy as np 
import math

from utils import *

n_of_timesteps = 32
len_of_predictions = 30
model_path = sys.argv[1]
dict_path = sys.argv[2]
file_path = sys.argv[3]
beam_search_k = int(sys.argv[4])
generated_song_name = sys.argv[5]
max_repetition = int(sys.argv[6]) if len(sys.argv) > 6 else len_of_predictions
tabu_list_length = int(sys.argv[7]) if len(sys.argv) > 7 else 0

def convert_to_midi(prediction_output, filename):
   
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                
                cn=int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
                
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        # pattern is a note
        else:
            
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 1
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)

def beam_search_generator(starting_data, model, len_of_songs, k):
    n_of_timesteps = len(starting_data[0])
    
    # starting k randomly generated states
    indices = np.random.randint(len(starting_data), size=k)
    # [prediction, random_music, score]
    sequences = [[[], starting_data[index], 0.0] for index in indices]
    # walk over each step in sequence

    for _ in range(len_of_songs):
        assert(len(sequences) == k)
        all_candidates = list()
        all_random_music = [sequence[1] for sequence in sequences]
        all_random_music = np.array(all_random_music).reshape(k, n_of_timesteps)
        prob = model.predict(all_random_music)
        # expand each current candidate
        for i in range(k):
            seq, random_music, score = sequences[i]
            for j in range(len(prob[i])):
                assert(len(random_music) == n_of_timesteps)
                if len(seq) >= max_repetition and all(note == j for note in seq[-max_repetition:]):
                    continue
                if tabu_list_length > 0 and any(note == j for note in seq[-tabu_list_length:]):
                    continue
                new_random_music = np.insert(random_music,len(random_music),j)
                new_random_music = new_random_music[1:]
                candidate = [seq + [j], new_random_music, score - math.log(prob[i][j])]
                all_candidates.append(candidate)

        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[2])
        # select k best
        sequences = ordered[:k]
    return sequences
    
def generate_song(model_path, model_dict_path, file, beam_search_k, timesteps, len_of_songs):
    model = load_model(model_path)
    note_to_num_dict = pickle.load( open( model_dict_path, "rb" ) )
    sample_song_notes = read_midi(file)

    assert len(sample_song_notes) > timesteps, "ERROR : sample song is empty!"

    sample_song_nums = np.array([note_to_num_dict[note_] for note_ in sample_song_notes if note_ in note_to_num_dict.keys()])
    
    starting_data = []
    
    for i in range(0, len(sample_song_nums) - timesteps, 1):
        starting_data.append(sample_song_nums[i:i + timesteps])
                    
    starting_data=np.array(starting_data)

    result = beam_search_generator(starting_data, model, len_of_songs, beam_search_k)
    num_to_note_dict = {v: k for k, v in note_to_num_dict.items()}
    
    return [num_to_note_dict[i] for i in result[0][0]]
    
np.random.seed(0)
generated_song = generate_song(model_path, dict_path, file_path, beam_search_k, n_of_timesteps, len_of_predictions)
convert_to_midi(generated_song, generated_song_name)