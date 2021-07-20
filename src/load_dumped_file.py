'''
    Using pickle to load dumped svr model configuration from the file
'''

import pickle



def load_dumped_file(path):
    '''
        Load svr model configuration from file
    '''
    with open(path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model
