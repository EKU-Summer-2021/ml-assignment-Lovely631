"""
    Using pickle to load dumped model configuration from the file
"""

import pickle


def load_dumped_file(path):
    """
        Load model configuration from file
    """
    with open(path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model
