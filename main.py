"""
    Main module
"""

from src.decision_tree import DecisionTree
from src.load_dumped_file import load_dumped_file
from src.make_directory import make_directory
from src.read_in import read_in


def dt_run():
    """
        Run decision tree class
    """
    dataset = read_in('https://raw.githubusercontent.com/EKU-Summer-2021/ml-assignment-Lovely631/master/data/heart.csv')
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
    print(load_results)

    decision_tree.plot_decision_tree()
    decision_tree.plot_confusion_matrix()


if __name__ == '__main__':
    dt_run()
