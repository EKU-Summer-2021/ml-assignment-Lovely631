'''
    Read csv data test
'''

import unittest

import pandas as pd

from src.read_in import read_in_csv


class CsvReadDataTest(unittest.TestCase):
    """
        CSV read in test
    """

    def setUp(self):
        self.dataset = read_in_csv(
            'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/heart.csv')

    def test_read_in_csv(self):
        """
            Unit test for csv read in instance
        """
        # given
        expected = True
        # when
        actual = isinstance(self.dataset, pd.DataFrame)
        # then
        self.assertEqual(expected, actual)

    def test_read_in_number_of_columns(self):
        """
            Unit test for csv read in number of columns
        """
        # given
        expected = 14
        # when
        actual = len(self.dataset.columns)
        # then
        self.assertEqual(expected, actual)

    def test_read_in_number_of_rows(self):
        """
            Unit test for csv read in number of rows
        """
        # given
        expected = 303
        # when
        actual = len(self.dataset)
        # then
        self.assertEqual(expected, actual)
