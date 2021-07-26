"""
    Load dumped file test class
"""

import unittest

from sklearn.model_selection import GridSearchCV

from src.load_dumped_file import load_dumped_file
from src.make_directory import make_directory
from src.read_in import read_in_csv
from src.reformat_avocado_dataset import reformat_avocado_dataset
from src.support_vector_machine import SupportVectorMachine


class LoadDumpedFileTest(unittest.TestCase):
    """
        Load dumped file test
    """

    def setUp(self):
        """
            Contains svm run for testing
        """
        self.dataset = read_in_csv(
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

    def test_load_dumped_file(self):
        """
            Unit test for dumped file
        """

        # given
        expected = True
        self.svm.best_estimator_from_grid_search_or_existing_load()
        path = self.svm._SupportVectorMachine__dump_results_of_grid_search()
        load_results = load_dumped_file(path)
        print(load_results)
        # when
        actual = isinstance(load_results, GridSearchCV)
        # then
        self.assertEqual(expected, actual)
