"""
    Multilayer perceptron regressor test
"""
import os
import unittest

import pandas as pd
from sklearn.neural_network import MLPRegressor

from src.make_directory import make_directory
from src.mlp_regressor import MlpRegressor
from src.read_in import read_in_csv
from src.reformat_avocado_dataset import reformat_avocado_dataset


class MlpRegressorTest(unittest.TestCase):
    """
        Multilayer perceptron regressor class test
    """

    def setUp(self):
        """
            Contains MLP regressor run for testing
        """
        dataset = read_in_csv(
            'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/avocado.csv')
        reformatted_dataset = reformat_avocado_dataset(dataset)

        param_grid = [{
            'hidden_layer_sizes': [(100, )],
            'activation': ['identity'],
            'solver': ['adam'],
            'alpha': [0.0001, ],
            'max_iter': [50],
        }
        ]

        path_to_mlp_regressor_directory = make_directory('mlp_regressor')
        self.mlp_regressor = MlpRegressor(reformatted_dataset, param_grid, path_to_mlp_regressor_directory)
        # mlp_regressor.split_dataset_to_train_and_test()
        self.mlp_regressor.best_estimator_from_grid_search_or_existing_load()

    def test_grid_search_without_path(self):
        """
            Grid search without path instance test
        """

        # given
        expected = True
        # when
        actual = isinstance(self.mlp_regressor.best_estimator_from_grid_search_or_existing_load(), MLPRegressor)
        # then
        self.assertEqual(expected, actual)

    def test_grid_search_with_path(self):
        """
            Grid search with path instance test
        """

        # given
        expected = True
        path = self.mlp_regressor._MlpRegressor__dump_results_of_grid_search()
        # when
        actual = isinstance(self.mlp_regressor.best_estimator_from_grid_search_or_existing_load(path), MLPRegressor)
        # then
        self.assertEqual(expected, actual)

    def test_dump_results_of_grid_search(self):
        """
            Dump results into a file test
        """

        # given
        expected = True
        path = self.mlp_regressor._MlpRegressor__dump_results_of_grid_search()
        # when
        actual = os.path.isfile(path)
        # then
        self.assertEqual(expected, actual)

    def test_save_best_grid_search_results(self):
        """
            Grid search best results save instance test
        """

        # given
        expected = True
        # when
        actual = isinstance(self.mlp_regressor._MlpRegressor__save_best_grid_search_results(), pd.DataFrame)
        # then
        self.assertEqual(expected, actual)

    def test_save_best_grid_search_result_columns(self):
        """
            Grid search best results number of columns test
        """

        # given
        expected = 15
        # when
        dataframe = self.mlp_regressor._MlpRegressor__save_best_grid_search_results()
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
        actual = isinstance(self.mlp_regressor.put_real_and_predicted_values_into_dataframe(), pd.DataFrame)
        # then
        self.assertEqual(expected, actual)

    def test_put_real_and_predicted_values_into_dataframe(self):
        """
            Dataframe which stores real and predicted y values number of columns test
        """

        # given
        expected = 2
        # when
        dataframe = self.mlp_regressor.put_real_and_predicted_values_into_dataframe()
        actual = len(dataframe.columns)
        # then
        self.assertEqual(expected, actual)
