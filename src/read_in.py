'''
    Read csv data
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing


def read_in(data):
    '''
        Read in function from csv
    '''
    pd.set_option("max_columns", None)
    pd.set_option('max_colwidth', None)
    pd.set_option("expand_frame_repr", False)
    dataset = pd.read_csv(data)
    reformatted_data = dataset.drop(['Index', 'Date', 'Small Bags', 'Large Bags', 'XLarge Bags'], axis=1)
    reformatted_data['Total Volume'] = reformatted_data['Total Volume'] - reformatted_data['Total Bags']
    reformatted_data['Type'] = np.where(reformatted_data['Type'] == 'conventional', 0, 1)
    # conventional 0 ; organic 1
    label_encoder = preprocessing.LabelEncoder()
    reformatted_data['Region'] = label_encoder.fit_transform(reformatted_data['Region'])
    # print(reformatted_data['Region'].unique())
    return reformatted_data
