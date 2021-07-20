'''
    Support vector machine module
'''

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class SupportVectorMachine:
    '''
        Support vector machine class
    '''

    def __init__(self, dataframe: pd.DataFrame, param_grid, path):
        self.dataframe = dataframe
        self.path = path
        self.sc_x = StandardScaler()
        self.sc_y = StandardScaler()
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_dataset_train_test()
        self.grid_svr: GridSearchCV
        self.best_estimated_result = self.grid_search(param_grid)

    def split_dataset_train_test(self):
        '''
            Splitting the dataframe into train and test
        '''

        x_columns = self.dataframe.iloc[:, 1:9]
        y_column = self.dataframe.iloc[:, 0].values
        x_scaled = self.sc_x.fit_transform(x_columns)
        y_scaled = self.sc_y.fit_transform(y_column.reshape(-1, 1))  # average price
        # print(x_columns)
        # print(y_column)
        # print(x_scaled[:, 0].shape, y_scaled.shape)
        # plt.scatter(x_scaled[:, 0], y_scaled)
        # plt.show()
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33)
        print(x_test.shape, x_test.shape, y_train.shape, y_test.shape)
        return x_train, x_test, y_train, y_test

    def grid_search(self, param_grid):
        '''
            Finds the best SVR model with GridSearchCv
        '''
        svr = SVR(verbose=True)
        grid_svr = GridSearchCV(svr, param_grid, cv=2, verbose=3)
        grid_svr.fit(self.x_train, self.y_train.ravel())
        self.grid_svr = grid_svr
        return self.grid_svr.best_estimator_

    def grid_search_best(self):
        '''
            Choosing the best result and creating a csv file
        '''
        results = pd.DataFrame(self.grid_svr.cv_results_)
        results = pd.DataFrame(results)
        results.to_csv(self.path + "/best_result_of_grid_search.csv")
        return results

    def dump_results(self):
        '''
            Using pickle to dump running results in a file
        '''
        filename = "all_result_of_grid_search"
        done_path = os.path.join(self.path, filename)
        with open(done_path, 'wb') as file:
            pickle.dump(self.best_estimated_result, file)
        return done_path

    def real_and_predicted_values(self):
        '''
            Creating dataframe which stores real and predicted y values
        '''
        y_prediction = self.best_estimated_result.predict(self.x_test)
        #print(y_prediction)
        y_prediction = self.sc_y.inverse_transform(y_prediction)
        #print(y_prediction)
        #print(self.y_test.shape)
        #print(a)
        real_predicted_values_dataframe = pd.DataFrame({
            'Real Values':  self.sc_y.inverse_transform(self.y_test.reshape(1, -1))[0],
            'Predicted Values': y_prediction})
        return real_predicted_values_dataframe

    def plot(self, dataframe):
        '''
            Plotting the x and y
        '''
        transformed = self.sc_x.inverse_transform(self.x_test)
        plt.scatter(transformed[:, 0], dataframe.iloc[:, 0])
        plt.title("Support Vector Machine - Total Volume and Real Value: y")
        plt.xlabel('Total Volume')
        plt.ylabel('Real Value of y_test')
        plt.savefig(self.path + "/SVM - Total Volume and Real Value.png")
        plt.show()

    def plot_line(self, dataframe: pd.DataFrame):
        '''
            Plotting the real y and the predicted y
        '''
        dataframe = dataframe.sort_values(by=['Real Values'], ascending=True)
        # print(dataframe)
        plt.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1])
        plt.title("Real Values and Predicted Values of Y")
        plt.xlabel('y_prediction')
        plt.ylabel('y_test')
        plt.savefig(self.path + "/y_prediction, y_test.png")
        plt.show()
