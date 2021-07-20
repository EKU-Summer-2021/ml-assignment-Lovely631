'''
    Function to make directories
'''

import datetime
import os


def make_directory(child_directory):
    '''
        Directory creation
    '''
    directory = os.getcwd()
    parent_dir = 'result'
    file_directory = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = "%s/%s/%s/%s" % (directory, parent_dir, child_directory, file_directory)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
