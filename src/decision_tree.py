"""
    Decision tree - classification
"""

import os
import pickle
from logging import getLogger

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import mean_squared_error, plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.load_dumped_file import load_dumped_file

logger = getLogger("logger")


class DecisionTree:
    """
        Decision tree class
    """

    def __init__(self, dataframe: pd.DataFrame, param_grid, path, model_path=None):
        self.dataframe = dataframe
        self.param_grid = param_grid
        self.path = path
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_dataset_to_train_and_test()
        self.grid_dt: GridSearchCV
        self.model = self.best_estimator_from_grid_search_or_existing_load(model_path)

    def split_dataset_to_train_and_test(self):
        """
            Splitting the dataframe into train and test
        """
        x_columns = self.dataframe.iloc[:, 0:-1]
        y_column = self.dataframe.iloc[:, -1].values
        logger.debug(x_columns)
        logger.debug(y_column)
        x_train, x_test, y_train, y_test = train_test_split(x_columns, y_column, test_size=0.33)
        logger.debug("%s%s%s%s" % (x_train.shape, x_test.shape, y_train.shape, y_test.shape))
        return x_train, x_test, y_train, y_test

    def best_estimator_from_grid_search_or_existing_load(self, path=None):
        """
            Gives back best estimator with grid search or load model
        """
        if path is not None:
            model = load_dumped_file(path)
        else:
            model = self.__grid_search_with_dt_model(self.param_grid)
            self.__dump_results_of_grid_search()
            self.__save_best_grid_search_results()
        return model.best_estimator_

    def __grid_search_with_dt_model(self, param_grid):
        """
            Finds the best decision tree model with GridSearchCv
        """
        decision_tree = DecisionTreeClassifier()
        grid_dt = GridSearchCV(decision_tree, param_grid, cv=2, verbose=3)
        grid_dt.fit(self.x_train, self.y_train)
        self.grid_dt = grid_dt
        return self.grid_dt

    def __dump_results_of_grid_search(self):
        """
            Using pickle to dump running results in a file
        """
        filename = "dumped_all_result_of_grid_search"
        done_path = os.path.join(self.path, filename)
        with open(done_path, 'wb') as file:
            pickle.dump(self.grid_dt, file)
        return done_path

    def __save_best_grid_search_results(self):
        """
            Choosing the best result and creating a csv file
        """
        results = pd.DataFrame(self.grid_dt.cv_results_)
        results = pd.DataFrame(results)
        results.to_csv(self.path + "/saved_best_result_of_grid_search.csv")
        return results

    def create_confusion_matrix(self, path=None):
        """
            Predict y values and create confusion matrix
        """
        if path is not None:
            loaded = load_dumped_file(path)
            y_prediction = loaded.predict(self.x_test)
        else:
            y_prediction = self.grid_dt.predict(self.x_test)
        confusion_matrix_result = confusion_matrix(self.y_test, y_prediction)
        print("MSE:", mean_squared_error(self.y_test, y_prediction))
        return confusion_matrix_result

    def plot_decision_tree(self):
        """
            Plot decision tree
        """
        plt.figure(figsize=(15, 10))
        tree.plot_tree(self.model, feature_names=self.x_train.columns,
                       class_names=['no', 'yes'],
                       filled=True)
        plt.savefig(self.path + "/decision_tree.png")
        plt.show()

    def plot_confusion_matrix(self):
        """
            Plot confusion matrix
        """
        plot_confusion_matrix(self.model, self.x_test, self.y_test)
        plt.savefig(self.path + "/confusion_matrix")
        plt.show()
