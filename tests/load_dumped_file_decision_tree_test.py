"""
    Load dumped file test class
"""

import unittest

from sklearn.tree import DecisionTreeClassifier

from src.decision_tree import DecisionTree
from src.load_dumped_file import load_dumped_file
from src.make_directory import make_directory
from src.read_in import read_in


class LoadDumpedFileTest(unittest.TestCase):
    """
        Load dumped file test
    """

    def test_load_dumped_file(self):
        """
            Unit test for dumped file
        """
        dataset = read_in(
            'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/heart.csv')
        print(dataset)

        param_grid = [{
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, 14, 15],
            'min_samples_split': [6, 10, 14, 15, 20],
            'min_samples_leaf': [7, 11, 15, 20, 25]
        }
        ]

        path_to_dt_directory = make_directory('dt')
        decision_tree = DecisionTree(dataset, param_grid, path_to_dt_directory)

        reach_file_path = decision_tree.dump_results()

        load_results = load_dumped_file(reach_file_path)

        # given
        expected = True
        # when
        actual = isinstance(load_results, DecisionTreeClassifier)
        # then
        self.assertEqual(expected, actual)
