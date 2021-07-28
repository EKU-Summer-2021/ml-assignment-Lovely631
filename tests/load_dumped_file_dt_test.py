"""
    Load dumped file test class
"""

import unittest

from sklearn.model_selection import GridSearchCV

from src.decision_tree import DecisionTree
from src.load_dumped_file import load_dumped_file
from src.make_directory import make_directory
from src.read_in import read_in_csv


class LoadDumpedFileTest(unittest.TestCase):
    """
        Load dumped file test
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

    def test_load_dumped_file(self):
        """
            Unit test for dumped file
        """

        # given
        expected = True
        self.decision_tree.best_estimator_from_grid_search_or_existing_load()
        path = self.decision_tree._DecisionTree__dump_results_of_grid_search()
        load_results = load_dumped_file(path)
        # print(load_results)
        # when
        actual = isinstance(load_results, GridSearchCV)
        # then
        self.assertEqual(expected, actual)
