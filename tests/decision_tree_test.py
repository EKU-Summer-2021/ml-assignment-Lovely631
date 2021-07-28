"""
    Decision tree test class - regression
"""
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.decision_tree import DecisionTree
from src.make_directory import make_directory
from src.read_in import read_in_csv


class DecisionTreeTest(unittest.TestCase):
    """
        Decision tree test
    """

    def setUp(self):
        """
            Contains decision tree run for testing
        """
        dataset = read_in_csv(
            'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/heart.csv')

        param_grid = [{
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, 14, 15],
            'min_samples_split': [6, 10, 14, 15, 20],
            'min_samples_leaf': [7, 11, 15, 20, 25]
        }
        ]

        path_to_dt_directory = make_directory('dt')
        self.decision_tree = DecisionTree(dataset, param_grid, path_to_dt_directory)
        self.decision_tree.best_estimator_from_grid_search_or_existing_load()

    def test_grid_search_without_path(self):
        """
            Grid search without path test
        """

        # given
        expected = True
        # when
        actual = isinstance(self.decision_tree.best_estimator_from_grid_search_or_existing_load(),
                            DecisionTreeClassifier)
        # then
        self.assertEqual(expected, actual)

    def test_grid_search_with_path(self):
        """
            Grid search with path test
        """

        # given
        expected = True
        path = self.decision_tree._DecisionTree__dump_results_of_grid_search()
        # when
        actual = isinstance(self.decision_tree.best_estimator_from_grid_search_or_existing_load(path),
                            DecisionTreeClassifier)
        # then
        self.assertEqual(expected, actual)

    def test_dump_results_of_grid_search(self):
        """
            Dump results into a file test
        """

        # given
        expected = True
        path = self.decision_tree._DecisionTree__dump_results_of_grid_search()
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
        actual = isinstance(self.decision_tree._DecisionTree__save_best_grid_search_results(), pd.DataFrame)
        # then
        self.assertEqual(expected, actual)

    def test_save_best_grid_search_result_columns(self):
        """
            Grid search best results number of columns test
        """

        # given
        expected = 14
        # when
        dataframe = self.decision_tree._DecisionTree__save_best_grid_search_results()
        actual = len(dataframe.columns)
        # then
        self.assertEqual(expected, actual)

    def test_create_confusion_matrix(self):
        """
            Predict y values and create confusion matrix (ndarray)
        """

        # given
        expected = True
        # when
        actual = isinstance(self.decision_tree.create_confusion_matrix(), np.ndarray)
        # then
        self.assertEqual(expected, actual)
