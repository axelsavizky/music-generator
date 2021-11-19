import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pickle
from feedforward import feedforward, geo_mean_overflow, load
import sys

import warnings
warnings.filterwarnings('ignore')

MIDIS_PATH = 'gwern/midis/Big_Data_Set/'
TEST_SET_LENGTH = 512

def calc_perplexity(model_path, model_dict_path, notes_array):
    result = feedforward(model_path, model_dict_path, notes_array)
    return geo_mean_overflow(result)


runs = [
    ("model_bigdataset_100_relu", "bigdataset_100_dict.pickle"),
    ("model_bigdataset_100_tanh", "bigdataset_100_dict.pickle"),
    ("model_bigdataset_100_sigmoid", "bigdataset_100_dict.pickle"),
    ("model_bigdataset_1000_relu", "bigdataset_1000_dict.pickle"),
    ("model_bigdataset_1000_tanh", "bigdataset_1000_dict.pickle"),
    ("model_bigdataset_1000_sigmoid", "bigdataset_1000_dict.pickle"),
    ("model_bigdataset_5000_relu", "bigdataset_5000_dict.pickle"),
    ("model_bigdataset_5000_tanh", "bigdataset_5000_dict.pickle"),
    ("model_bigdataset_5000_sigmoid", "bigdataset_5000_dict.pickle"),
    ("model_bigdataset_10000_relu", "bigdataset_10000_dict.pickle"),
    ("model_bigdataset_10000_tanh", "bigdataset_10000_dict.pickle"),
    ("model_bigdataset_10000_sigmoid", "bigdataset_10000_dict.pickle"),
    ("model_bigdataset_20000_relu", "bigdataset_20000_dict.pickle"),
    ("model_bigdataset_20000_tanh", "bigdataset_20000_dict.pickle"),
    ("model_bigdataset_20000_sigmoid", "bigdataset_20000_dict.pickle"),
]


if __name__ == "__main__":
    TEST_SET_LENGTH = int(sys.argv[1])
    print("#test_set_length: " + str(TEST_SET_LENGTH))
    results = {}
    notes_array = load(MIDIS_PATH, TEST_SET_LENGTH)
    for model, dict_notes in runs:
        results[model] = calc_perplexity(model, dict_notes, notes_array)

    print(list(results.items()))