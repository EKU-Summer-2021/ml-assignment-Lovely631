"""
    Support vector machine test class - regression
"""
import os
import unittest

import pandas as pd
from sklearn.svm import SVR

from src.make_directory import make_directory
from src.read_in import read_in_csv
from src.reformat_avocado_dataset import reformat_avocado_dataset
from src.support_vector_machine import SupportVectorMachine


class SupportVectorMachineTest(unittest.TestCase):
    """
        Support vector machine class test
    """

    def setUp(self):
        """
            Contains svm run for testing
        """
        dataset = read_in_csv(
            'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/avocado.csv')
        reformatted_dataset = reformat_avocado_dataset(dataset)
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
            Grid search without path test
        """

        # given
        expected = True
        # when
        actual = isinstance(self.svm.best_estimator_from_grid_search_or_existing_load(), SVR)
        # then
        self.assertEqual(expected, actual)

    def test_grid_search_with_path(self):
        """
            Grid search with path test
        """

        # given
        expected = True
        path = self.svm._SupportVectorMachine__dump_results_of_grid_search()
        # when
        actual = isinstance(self.svm.best_estimator_from_grid_search_or_existing_load(path), SVR)
        # then
        self.assertEqual(expected, actual)

    def test_dump_results_of_grid_search(self):
        """
            Dump results into a file test
        """

        # given
        expected = True
        path = self.svm._SupportVectorMachine__dump_results_of_grid_search()
        # when
        actual = os.path.isfile(path)
        # then
        self.assertEqual(expected, actual)

    def test_save_best_grid_search_results(self):
        """
            Grid search best results save test
        """

        # given
        expected = True
        # when
        actual = isinstance(self.svm._SupportVectorMachine__save_best_grid_search_results(), pd.DataFrame)
        # then
        self.assertEqual(expected, actual)

    def test_save_best_grid_search_result_rows(self):
        """
            Grid search best results number of columns test
        """

        # given
        expected = 14
        # when
        dataframe = self.svm._SupportVectorMachine__save_best_grid_search_results()
        actual = len(dataframe.columns)
        # then
        self.assertEqual(expected, actual)

    def test_put_real_and_predicted_values_into_dataframe(self):
        """
            Dataframe which stores real and predicted y values test
        """

        # given
        expected = True
        # when
        actual = isinstance(self.svm.put_real_and_predicted_values_into_dataframe(), pd.DataFrame)
        # then
        self.assertEqual(expected, actual)

    def test_put_real_and_predicted_values_into_dataframe(self):
        """
            Dataframe which stores real and predicted y values number of columns test
        """

        # given
        expected = 2
        # when
        dataframe = self.svm.put_real_and_predicted_values_into_dataframe()
        actual = len(dataframe.columns)
        # then
        self.assertEqual(expected, actual)
