"""
    Read csv data test
"""

import unittest

from sklearn.svm import SVR

from src.make_directory import make_directory
from src.read_in import read_in
from src.reformat_avocado_dataset import reformat_avocado_dataset
from src.support_vector_machine import SupportVectorMachine


class SupportVectorMachineTest(unittest.TestCase):
    """
        Support vector machine class test
    """

    def test_svm(self):
        """
            SVM test
        """

        dataset = read_in(
            'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/avocado.csv')
        reformatted_dataset = reformat_avocado_dataset(dataset)
        param_grid = [{
            'kernel': ['rbf'],
            'gamma': [1],
            'C': [0.1],
            'epsilon': [0.2]
        }
        ]

        path_to_svm_directory = make_directory('svm')
        svm = SupportVectorMachine(reformatted_dataset, param_grid, path_to_svm_directory)

        svm.split_dataset_train_test()
        svm.grid_search(param_grid)

        # given
        expected = True
        # when
        actual = isinstance(svm.grid_search(param_grid), SVR)
        # then
        self.assertEqual(expected, actual)
