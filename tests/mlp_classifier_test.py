"""
    Decision tree test class - regression
"""
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.make_directory import make_directory
from src.mlp_classifier import MlpClassifier
from src.read_in import read_in_csv


class MlpClassifierTest(unittest.TestCase):
    """
        Multilayer perceptron classifier test
    """

    def setUp(self):
        """
            Contains MLP classifier run for testing
        """
        dataset = read_in_csv(
            'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/heart.csv')

        param_grid = [{
            'hidden_layer_sizes': [(100, 100, 100)],
            'activation': ['identity', 'relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.05],
            'max_iter': [500],
        }
        ]

        path_to_mlp_classifier_directory = make_directory('mlp_classifier')
        self.mlp_classifier = MlpClassifier(dataset, param_grid, path_to_mlp_classifier_directory)
        self.mlp_classifier.best_estimator_from_grid_search_or_existing_load()

    def test_grid_search_without_path(self):
        """
            Grid search without path test
        """

        # given
        expected = True
        # when
        actual = isinstance(self.mlp_classifier.best_estimator_from_grid_search_or_existing_load(),
                            MLPClassifier)
        # then
        self.assertEqual(expected, actual)

    def test_grid_search_with_path(self):
        """
            Grid search with path test
        """

        # given
        expected = True
        path = self.mlp_classifier._MlpClassifier__dump_results_of_grid_search()
        # when
        actual = isinstance(self.mlp_classifier.best_estimator_from_grid_search_or_existing_load(path),
                            MLPClassifier)
        # then
        self.assertEqual(expected, actual)

    def test_dump_results_of_grid_search(self):
        """
            Dump results into a file test
        """

        # given
        expected = True
        path = self.mlp_classifier._MlpClassifier__dump_results_of_grid_search()
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
        actual = isinstance(self.mlp_classifier._MlpClassifier__save_best_grid_search_results(), pd.DataFrame)
        # then
        self.assertEqual(expected, actual)

    def test_create_confusion_matrix(self):
        """
            Predict y values and create confusion matrix (ndarray)
        """

        # given
        expected = True
        # when
        actual = isinstance(self.mlp_classifier.create_confusion_matrix(), np.ndarray)
        # then
        self.assertEqual(expected, actual)
