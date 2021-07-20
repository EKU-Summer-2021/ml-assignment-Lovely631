'''
Main modul
'''

from src.read_in import read_in
from src.make_directory import make_directory
from src.support_vector_machine import SupportVectorMachine
from src.load_dumped_file import load_dumped_file

if __name__ == '__main__':
    dataset = read_in(
        'https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/avocado.csv')
    # print(dataset)

    param_grid = [{
        'kernel': ['rbf', 'sigmoid'],
        'gamma': [1, 0.1, 0.001],
        'C': [0.1, 1, 10, 700, 1000],
        'epsilon': [0.2, 0.1]
    }
    ]

    param_grid_1 = [{
        'kernel': [ 'rbf', 'sigmoid'],
        'gamma': [0.001, 0.0001],
        'C': [300, 1000],
        'epsilon': [0.2, 0.1]
        }
    ]

    path_to_svm_directory = make_directory('svm')
    svm = SupportVectorMachine(dataset, param_grid, path_to_svm_directory)
    # svm.split_datatest_train_test()

    svm.grid_search_best()
    reach_file_path = svm.dump_results()

    load_results = load_dumped_file(reach_file_path)
    print(load_results)

    dataframe_of_real_and_predicted_values = svm.real_and_predicted_values()
    print(dataframe_of_real_and_predicted_values)

    svm.plot(dataframe_of_real_and_predicted_values)
    svm.plot_line(dataframe_of_real_and_predicted_values)
