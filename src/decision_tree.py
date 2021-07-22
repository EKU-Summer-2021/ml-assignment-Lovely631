"""
    Decision tree - classification
"""
import os
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    """
        Decision tree class
    """
    def __init__(self, dataframe: pd.DataFrame, param_grid, path):
        self.dataframe = dataframe
        self.path = path
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_dataset_train_test()
        self.grid_svr: GridSearchCV
        self.best_estimated_result = self.grid_search(param_grid)

    def split_dataset_train_test(self):
        """
            Splitting the dataframe into train and test
        """
        x_columns = self.dataframe.iloc[:, 0:-1]
        y_column = self.dataframe.iloc[:, -1]  # heart disease
        x_train, x_test, y_train, y_test = train_test_split(x_columns, y_column, test_size=0.33)
        print(x_test.shape, x_test.shape, y_train.shape, y_test.shape)
        return x_train, x_test, y_train, y_test

    def grid_search(self, param_grid):
        """
            Finds the best DT model with GridSearchCv
        """
        decision_tree = DecisionTreeClassifier()
        grid_dt = GridSearchCV(decision_tree, param_grid, cv=2, verbose=3)
        grid_dt.fit(self.x_train, self.y_train)
        self.grid_dt = grid_dt
        return self.grid_dt.best_estimator_

    def grid_search_best(self):
        """
            Choosing the best result and creating a csv file
        """
        results = pd.DataFrame(self.grid_dt.cv_results_)
        results = pd.DataFrame(results)
        results.to_csv(self.path + "/best_result_of_decision_tree.csv")
        return results

    def dump_results(self):
        """
            Using pickle to dump running results in a file
        """
        filename = "all_result_of_grid_search"
        done_path = os.path.join(self.path, filename)
        with open(done_path, 'wb') as file:
            pickle.dump(self.best_estimated_result, file)
        return done_path

    def prediction(self):
        """
            Predict y values and create confusion matrix
        """
        y_prediction = self.best_estimated_result.predict(self.x_test)
        confusion_matrix_result = confusion_matrix(self.y_test, y_prediction)
        return confusion_matrix_result

    def plot_decision_tree(self):
        """
            Plot decision tree
        """
        plt.figure(figsize=(15, 10))
        tree.plot_tree(self.best_estimated_result, feature_names=self.x_train.columns, class_names=['no', 'yes'],
                       filled=True)
        plt.savefig(self.path + "/decision_tree.png")
        plt.show()

    def plot_confusion_matrix(self):
        """
            Plot confusion matrix
        """
        plot_confusion_matrix(self.best_estimated_result, self.x_test, self.y_test)
        plt.savefig(self.path + "/confusion_matrix")
        plt.show()
