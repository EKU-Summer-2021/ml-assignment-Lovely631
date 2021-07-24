"""
    Support vector machine test class - regression
"""
import os
import unittest
from unittest import mock

from sklearn.svm import SVR
from src.make_directory import make_directory
from src.read_in import read_in
from src.reformat_avocado_dataset import reformat_avocado_dataset
from src.support_vector_machine import SupportVectorMachine


class MainCode:
    """
        Contains svm run for testing
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
    reach_file_path = svm.dump_results()
    dataframe_of_real_and_predicted_values = svm.real_and_predicted_values()

    svm.plot(dataframe_of_real_and_predicted_values)
    svm.plot_line(dataframe_of_real_and_predicted_values)


class SupportVectorMachineTest(unittest.TestCase):
    """
        Support vector machine test
    """

    def test_svm(self):
        """
            SVM test
        """

        main_code = MainCode()

        # given
        expected = True
        # when
        actual = isinstance(main_code.svm.grid_search(main_code.param_grid), SVR)
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

    def test_plot(self):
        """
            Plot x and y test
        """

        with mock.patch("support_vector_machine_test.MainCode.svm.plot") as mock_my_method:
            MainCode().svm.plot(MainCode.dataframe_of_real_and_predicted_values)
            mock_my_method.assert_called_once()

    def test_plot_line(self):
        """
            Plot real y and predicted y test
        """

        with mock.patch("support_vector_machine_test.MainCode.svm.plot_line") as mock_my_method:
            MainCode().svm.plot_line(MainCode.dataframe_of_real_and_predicted_values)
            mock_my_method.assert_called_once()
