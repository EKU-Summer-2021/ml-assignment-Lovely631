"""
    Load dumped file test class
"""

import unittest

from sklearn.svm import SVR

from src.load_dumped_file import load_dumped_file
from src.make_directory import make_directory
from src.read_in import read_in
from src.support_vector_machine_with_regression.reformat_avocado_dataset import reformat_avocado_dataset
from src.support_vector_machine_with_regression.support_vector_machine import SupportVectorMachine


class LoadDumpedFileTest(unittest.TestCase):
    """
        Load dumped file test
    """

    def test_load_dumped_file(self):
        """
            Unit test for dumped file
        """
        dataset = read_in(
            'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/avocado.csv')
        reformatted_dataset = reformat_avocado_dataset(dataset)
        print(dataset)

        param_grid = [{
            'kernel': ['rbf'],
            'gamma': [1],
            'C': [0.1],
            'epsilon': [0.2]
        }
        ]

        path_to_svm_directory = make_directory('svm')
        svm = SupportVectorMachine(reformatted_dataset, param_grid, path_to_svm_directory)

        reach_file_path = svm.dump_results()

        load_results = load_dumped_file(reach_file_path)

        # given
        expected = True
        # when
        actual = isinstance(load_results, SVR)
        # then
        self.assertEqual(expected, actual)
