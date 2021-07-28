"""
    Main module
"""

import pandas as pd

from src.decision_tree import DecisionTree
from src.make_directory import make_directory
from src.mlp_classifier import MlpClassifier
from src.mlp_regressor import MlpRegressor
from src.read_in import read_in_csv
# from src.reformat_mlp_regressor_dataset import reformat_mlp_regressor_dataset
from src.reformat_avocado_dataset import reformat_avocado_dataset
from src.support_vector_machine import SupportVectorMachine

pd.set_option("max_columns", None)
pd.set_option("max_colwidth", None)
pd.set_option("expand_frame_repr", False)


def support_vector_machine_with_regression_run():
    """
        Run support vector machine class
    """

    dataset = read_in_csv(
        'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/avocado.csv')
    reformatted_dataset = reformat_avocado_dataset(dataset)
    # print(dataset)

    param_grid = [{
        'kernel': ['rbf', 'sigmoid'],
        'gamma': [1, 0.1, 0.001],
        'C': [0.1, 1, 10, 700, 1000],
        'epsilon': [0.2, 0.1]
    }
    ]
    path_to_svm_directory = make_directory('svm')
    '''
        Run without path
    '''

    # svm = SupportVectorMachine(reformatted_dataset, param_grid, path_to_svm_directory)
    # svm.split_dataset_to_train_and_test()
    # svm.best_estimator_from_grid_search_or_existing_load()
    # dataframe_of_real_and_predicted_values = svm.put_real_and_predicted_values_into_dataframe()
    '''
        Run with path
    '''

    path = 'result/svm/2021-07-28_08-14-20/dumped_all_result_of_grid_search'
    svm = SupportVectorMachine(reformatted_dataset, param_grid, path_to_svm_directory, path)
    svm.best_estimator_from_grid_search_or_existing_load(path)
    dataframe_of_real_and_predicted_values = svm.put_real_and_predicted_values_into_dataframe(path)

    svm.plot_total_volume_and_real_value(dataframe_of_real_and_predicted_values)
    svm.plot_real_and_predicted_value(dataframe_of_real_and_predicted_values)


def decision_tree_with_classification_run():
    """
        Run decision tree class
    """

    dataset = read_in_csv(
        'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/heart.csv')
    # print(dataset)

    param_grid = [{
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10, 14, 15],
        'min_samples_split': [6, 10, 14, 15, 20],
        'min_samples_leaf': [7, 11, 15, 20, 25]
    }
    ]
    path_to_dt_directory = make_directory('dt')
    '''
        Run without path
    '''

    # decision_tree = DecisionTree(dataset, param_grid, path_to_dt_directory)
    # decision_tree.split_dataset_to_train_and_test()
    # decision_tree.best_estimator_from_grid_search_or_existing_load()
    # confusion_matrix = decision_tree.create_confusion_matrix()
    '''
        Run with path
    '''

    path = 'result/dt/2021-07-28_08-51-42/dumped_all_result_of_grid_search'
    decision_tree = DecisionTree(dataset, param_grid, path_to_dt_directory, path)
    decision_tree.best_estimator_from_grid_search_or_existing_load(path)
    confusion_matrix = decision_tree.create_confusion_matrix(path)

    print('Confusion matrix:\n', confusion_matrix)

    decision_tree.plot_decision_tree()
    decision_tree.plot_confusion_matrix()


def multilayer_perceptron_regressor_run():
    """
        Multilayer perceptron regression class
    """

    dataset = read_in_csv(
        'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/avocado.csv')
    reformatted_dataset = reformat_avocado_dataset(dataset)

    # dataset = read_in_csv('https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/water_potability.csv')
    # reformatted_dataset = reformat_mlp_regressor_dataset(dataset)
    # print(dataset)

    param_grid = [{
        'hidden_layer_sizes': [(400, 500, 500, 500, 400)],
        'activation': ['tanh', 'identity', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'max_iter': [500],
    }
    ]

    path_to_mlp_regressor_directory = make_directory('mlp_regressor')
    '''
        Run without path
    '''

    mlp_regressor = MlpRegressor(reformatted_dataset, param_grid, path_to_mlp_regressor_directory)
    # mlp_regressor.split_dataset_to_train_and_test()
    mlp_regressor.best_estimator_from_grid_search_or_existing_load()
    dataframe_of_real_and_predicted_values = mlp_regressor.put_real_and_predicted_values_into_dataframe()
    '''
        Run with path
    '''

    # path = ''
    # mlp_regressor = SupportVectorMachine(reformatted_dataset, param_grid, path_to_svm_directory, path)
    # mlp_regressor.best_estimator_from_grid_search_or_existing_load(path)
    # dataframe_of_real_and_predicted_values = mlp_regressor.put_real_and_predicted_values_into_dataframe(path)

    mlp_regressor.plot_total_volume_and_real_value(dataframe_of_real_and_predicted_values)
    mlp_regressor.plot_real_and_predicted_value(dataframe_of_real_and_predicted_values)


def multilayer_perceptrom_classifier_run():
    '''
        Multilayer perceptron classifier class
    '''

    dataset = read_in_csv(
        'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/heart.csv')

    param_grid = [{
        'hidden_layer_sizes': [(100, 200, 200, 100)],
        'activation': ['tanh', 'identity', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'max_iter': [1000],
    }
    ]

    path_to_mlp_regressor_directory = make_directory('mlp_regressor')
    '''
        Run without path
    '''

    mlp_classifier = MlpClassifier(dataset, param_grid, path_to_mlp_regressor_directory)
    # mlp_classifier.split_dataset_to_train_and_test()
    mlp_classifier.best_estimator_from_grid_search_or_existing_load()
    mlp_classifier.create_confusion_matrix()

    '''
        Run with path
    '''

    # path = ''
    # mlp_classifier = SupportVectorMachine(reformatted_dataset, param_grid, path_to_svm_directory, path)
    # mlp_classifier.best_estimator_from_grid_search_or_existing_load(path)
    # mlp_classifier.create_confusion_matrix(path)

    mlp_classifier.plot_confusion_matrix()


if __name__ == '__main__':
    support_vector_machine_with_regression_run()
    decision_tree_with_classification_run()
    multilayer_perceptron_regressor_run()
    multilayer_perceptrom_classifier_run()
