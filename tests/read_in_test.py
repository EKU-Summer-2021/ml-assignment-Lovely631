'''
    Read csv data test
'''

import unittest

import pandas as pd

from src.read_in import read_in


class CsvReadDataTest(unittest.TestCase):
    """
        CSV read in test
    """

    def test_read_csv(self):
        """
            Unit test for csv read in
        """
        # given
        expected = True
        # when
        actual = isinstance(
            read_in('https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/avocado.csv')
            , pd.DataFrame)
        # then
        self.assertEqual(expected, actual)
