"""
    Support vector machine - regression
"""

import os
import pickle
from logging import getLogger

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from src.load_dumped_file import load_dumped_file

logger = getLogger("logger")


class MlpRegressor:
    """
        Multilayer perceptron regressor class
    """

    def __init__(self, dataframe: pd.DataFrame, param_grid, path, model_path=None):
        self.dataframe = dataframe
        self.param_grid = param_grid
        self.path = path
        self.sc_x = MinMaxScaler()
        self.sc_y = MinMaxScaler()
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_dataset_to_train_and_test()
        self.grid_mlp_classifier: GridSearchCV
        self.model = self.best_estimator_from_grid_search_or_existing_load(model_path)

    def split_dataset_to_train_and_test(self):
        """
            Splitting the dataframe into train and test
        """
        x_columns = self.dataframe.iloc[:, 0:-1]
        y_column = self.dataframe.iloc[:, -1].values
        x_scaled = self.sc_x.fit_transform(x_columns)
        y_scaled = self.sc_y.fit_transform(y_column.reshape(-1, 1))
        logger.debug(x_columns)
        logger.debug(y_column)
        logger.debug("%s/%s" % (x_scaled[:, 0].shape, y_scaled.shape))
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33)
        logger.debug("%s/%s/%s/%s" % (x_train.shape, x_test.shape, y_train.shape, y_test.shape))
        return x_train, x_test, y_train, y_test

    def best_estimator_from_grid_search_or_existing_load(self, path=None):
        """
            Gives back best estimator with grid search or load model
        """
        if path is not None:
            model = load_dumped_file(path)
        else:
            model = self.__grid_search_with_mlp_regressor_model(self.param_grid)
            self.__dump_results_of_grid_search()
            self.__save_best_grid_search_results()
        return model.best_estimator_

    def __grid_search_with_mlp_regressor_model(self, param_grid):
        """
            Finds the best Mlp regressor model with GridSearchCv
        """
        mlp_regressor = MLPRegressor(verbose=True)
        grid_mlp_regressor = GridSearchCV(mlp_regressor, param_grid, cv=2, verbose=3)
        grid_mlp_regressor.fit(self.x_train, self.y_train.ravel())
        self.grid_mlp_classifier = grid_mlp_regressor
        return self.grid_mlp_classifier

    def __dump_results_of_grid_search(self):
        """
            Using pickle to dump running results in a file
        """
        filename = "dumped_result_of_grid_search"
        done_path = os.path.join(self.path, filename)
        with open(done_path, 'wb') as file:
            pickle.dump(self.grid_mlp_classifier, file)
        return done_path

    def __save_best_grid_search_results(self):
        """
            Choosing the best result and creating a csv file
        """
        results = pd.DataFrame(self.grid_mlp_classifier.cv_results_)
        results = pd.DataFrame(results)
        results.to_csv(self.path + "/saved_best_result_of_grid_search.csv")
        logger.debug('columns')
        logger.debug(len(results.columns))
        logger.debug(results)
        return results

    def put_real_and_predicted_values_into_dataframe(self, path=None):
        """
            Creating dataframe which stores real and predicted y values
        """
        if path is not None:
            loaded = load_dumped_file(path)
            y_prediction = loaded.predict(self.x_test)
        else:
            y_prediction = self.grid_mlp_classifier.predict(self.x_test)
            logger.debug(y_prediction)
            y_prediction = self.sc_y.inverse_transform([y_prediction])
        y = self.sc_y.inverse_transform(self.y_test.reshape(1, -1))[0]
        real_predicted_values_dataframe = pd.DataFrame({
            'Real Values': y,
            'Predicted Values': y_prediction[0]})
        logger.debug('columns')
        logger.debug(len(real_predicted_values_dataframe.columns))
        logger.debug(real_predicted_values_dataframe)
        print("MSE:", mean_squared_error(y, y_prediction[0]))
        print(self.model.score)
        return real_predicted_values_dataframe

    def plot_total_volume_and_real_value(self, dataframe: pd.DataFrame):
        """
            Plotting the x and y
        """
        transformed = self.sc_x.inverse_transform(self.x_test)
        plt.scatter(transformed[:, 0], dataframe.iloc[:, 0])
        plt.title("Support Vector Machine - Total Volume and Real Value: y")
        plt.xlabel('Total Volume')
        plt.ylabel('Real Value of y_test')
        plt.savefig(self.path + "/SVM - Total Volume and Real Value.png")
        plt.show()

    def plot_real_and_predicted_value(self, dataframe: pd.DataFrame):
        """
            Plotting the real y and the predicted y
        """
        dataframe = dataframe.sort_values(by=['Real Values'], ascending=True)
        logger.debug(dataframe)
        plt.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], cmap='autumn')
        plt.title("Real Values and Predicted Values of Y")
        plt.xlabel('y_prediction')
        plt.ylabel('y_test')
        plt.savefig(self.path + "/y_prediction, y_test.png")
        plt.show()
