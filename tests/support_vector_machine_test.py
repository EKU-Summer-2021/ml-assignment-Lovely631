"""
    Support vector machine test class - regression
"""
import os
import sys
import unittest

import pandas as pd
from sklearn.svm import SVR

from src.make_directory import make_directory
from src.read_in import read_in
from src.reformat_avocado_dataset import reformat_avocado_dataset
from src.support_vector_machine import SupportVectorMachine


class SupportVectorMachineTest(unittest.TestCase):
    """
        Support vector machine test
    """

    def setUp(self):
        """
            Contains svm run for testing
        """
        self.dataset = read_in(
            'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/avocado.csv')
        reformatted_dataset = reformat_avocado_dataset(self.dataset)
        param_grid = [{
            'kernel': ['rbf'],
            'gamma': [1],
            'C': [0.1],
            'epsilon': [0.2]
        }
        ]

        self.path_to_svm_directory = make_directory('svm')
        self.svm = SupportVectorMachine(reformatted_dataset, param_grid, self.path_to_svm_directory)
        self.svm.best_estimator_from_grid_search_or_existing_load()

    def test_grid_search_without_path(self):
        """
            SVM test
        """

        # given
        expected = True
        # when
        actual = isinstance(self.svm.best_estimator_from_grid_search_or_existing_load(), SVR)
        # then
        self.assertEqual(expected, actual)

    @unittest.skipUnless(sys.platform.startswith('win'), 'This test only suppose to run on Windows.')
    def test_grid_search_with_path(self):
        """
            SVM test
        """

        # given
        expected = True
        path = \
            "D:/Codes/pycharm_pipenv/ml-assignment-Lovely631/result/svm/2021-07-26_16-01-18/all_result_of_grid_search"
        # when
        actual = isinstance(self.svm.best_estimator_from_grid_search_or_existing_load(path), SVR)
        # then
        self.assertEqual(expected, actual)

    def test_dump_results(self):
        """
            Dump results into a file test
        """

        # given
        expected = True
        path = self.svm._SupportVectorMachine__dump_results()
        # when
        actual = os.path.isfile(path)
        # then
        self.assertEqual(expected, actual)

    def test_grid_search_save_best(self):
        """
            Grid search best results save test
        """

        # given
        expected = True
        # when
        actual = isinstance(self.svm._SupportVectorMachine__grid_search_save_best(), pd.DataFrame)
        # then
        self.assertEqual(expected, actual)

    def test_real_and_predicted_values(self):
        """
            Dataframe which stores real and predicted y values
        """

        # given
        expected = True
        # when
        actual = isinstance(self.svm.real_and_predicted_values(), pd.DataFrame)
        # then
        self.assertEqual(expected, actual)
