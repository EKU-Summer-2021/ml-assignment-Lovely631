"""
    Decision tree test class - classification
"""
import os
import unittest

from sklearn.tree import DecisionTreeClassifier
from src.decision_tree import DecisionTree
from src.make_directory import make_directory
from src.read_in import read_in


class DecisionTreeTest(unittest.TestCase):
    """
        Decision tree test
    """

    def test_dt(self):
        """
            DT test
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

        decision_tree.split_dataset_train_test()
        decision_tree.grid_search(param_grid)

        # given
        expected = True
        # when
        actual = isinstance(decision_tree.grid_search(param_grid), DecisionTreeClassifier)
        # then
        self.assertEqual(expected, actual)

    def test_dump_results(self):
        """
            Dump results into a file test
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

        # given
        expected = True
        # when
        actual = os.path.isfile(reach_file_path)
        # then
        self.assertEqual(expected, actual)
