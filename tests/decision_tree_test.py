"""
    Decision tree test class - classification
"""
import os
import unittest
from unittest import mock

from sklearn.tree import DecisionTreeClassifier

from src.decision_tree import DecisionTree
from src.load_dumped_file import load_dumped_file
from src.make_directory import make_directory
from src.read_in import read_in


class MainCode:
    """
        Contains decision tree run for testing
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

    decision_tree.split_dataset_train_test()

    decision_tree.grid_search_best()
    reach_file_path = decision_tree.dump_results()

    load_results = load_dumped_file(reach_file_path)

    decision_tree.plot_decision_tree()
    decision_tree.plot_confusion_matrix()


class DecisionTreeTest(unittest.TestCase):
    """
        Decision tree test
    """

    def test_dt(self):
        """
            DT test
        """

        main_code = MainCode()

        # given
        expected = True
        # when
        actual = isinstance(main_code.decision_tree.grid_search(main_code.param_grid), DecisionTreeClassifier)
        # then
        self.assertEqual(expected, actual)

    def test_dump_results(self):
        """
            Dump results into a file test
        """

        main_code = MainCode()

        # given
        expected = True
        # when
        actual = os.path.isfile(main_code.reach_file_path)
        # then
        self.assertEqual(expected, actual)

    def test_plot_decision_tree(self):
        """
            Plot x and y test
        """

        with mock.patch("decision_tree_test.MainCode.decision_tree.plot_decision_tree") as mock_my_method:
            MainCode().decision_tree.plot_decision_tree()
            mock_my_method.assert_called_once()

    def test_plot_line(self):
        """
            Plot real y and predicted y test
        """

        with mock.patch("decision_tree_test.MainCode.decision_tree.plot_confusion_matrix") as mock_my_method:
            MainCode().decision_tree.plot_confusion_matrix()
            mock_my_method.assert_called_once()
