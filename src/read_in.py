"""
    Read csv data
"""

import pandas as pd


def read_in(data):
    """
        Read in function from csv
    """
    pd.set_option("max_columns", None)
    pd.set_option("max_colwidth", None)
    pd.set_option("expand_frame_repr", False)
    dataset = pd.read_csv(data)
    return dataset
