"""
    Read csv data
"""

import pandas as pd


def read_in(data):
    """
        Read in function from csv
    """
    dataset = pd.read_csv(data)
    return dataset
