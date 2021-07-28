"""
    Load dumped file test class
"""

import unittest

from sklearn.model_selection import GridSearchCV

from src.load_dumped_file import load_dumped_file
from src.make_directory import make_directory
from src.mlp_classifier import MlpClassifier
from src.read_in import read_in_csv


class LoadDumpedFileTest(unittest.TestCase):
    """
        Load dumped file test
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

    def test_load_dumped_file(self):
        """
            Unit test for dumped file
        """

        # given
        expected = True
        self.mlp_classifier.best_estimator_from_grid_search_or_existing_load()
        path = self.mlp_classifier._MlpClassifier__dump_results_of_grid_search()
        load_results = load_dumped_file(path)
        # print(load_results)
        # when
        actual = isinstance(load_results, GridSearchCV)
        # then
        self.assertEqual(expected, actual)
