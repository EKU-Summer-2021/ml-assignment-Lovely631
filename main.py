"""
    Main module
"""

import pandas as pd

from src.make_directory import make_directory
from src.read_in import read_in_csv
from src.reformat_avocado_dataset import reformat_avocado_dataset
from src.support_vector_machine import SupportVectorMachine


def svm_run():
    """
        Run support vector machine class
    """

    pd.set_option("max_columns", None)
    pd.set_option("max_colwidth", None)
    pd.set_option("expand_frame_repr", False)

    dataset = read_in_csv(
        'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/avocado.csv')
    reformatted_dataset = reformat_avocado_dataset(dataset)
    print(dataset)

    param_grid = [{
        'kernel': ['rbf', 'sigmoid'],
        'gamma': [1, 0.1, 0.001],
        'C': [0.1, 1, 10, 700, 1000],
        'epsilon': [0.2, 0.1]
    }
    ]

    path_to_svm_directory = make_directory('svm')
    svm = SupportVectorMachine(reformatted_dataset, param_grid, path_to_svm_directory)
    # svm.split_dataset_train_test()

    path = 'result/svm/2021-07-26_16-01-18/all_result_of_grid_search'
    # svm.best_estimator_from_grid_search_or_existing_load()
    svm.best_estimator_from_grid_search_or_existing_load(path)

    # dataframe_of_real_and_predicted_values = svm.real_and_predicted_values()
    dataframe_of_real_and_predicted_values = svm.put_real_and_predicted_values_into_dataframe(path)
    svm.plot_total_volume_and_real_value(dataframe_of_real_and_predicted_values)
    svm.plot_real_and_predicted_value(dataframe_of_real_and_predicted_values)


if __name__ == '__main__':
    svm_run()
