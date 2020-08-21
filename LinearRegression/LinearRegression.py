import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, x, y):
        """
        Parameters
        ----------
        self.w : numpy array
            Theta's values
        self.cost : array
            Keep track of values of cost
        self.iteration : array
            Array with the number of iterations
        self.X : pandas Dataframe
            Contain the features
        self.y : pandas Dataframe
            Contain the labels
        self.X_n : pandas Dataframe
            Contain the normalized features
        self.y_n : pandas Dataframe
            Contain the normalized labels
        """

        self.X = np.array(x)
        self.y = np.array(y)
        self.X_n = np.array(x)
        self.y_n = np.array(y)
        self.m_data = len(self.X)
        self.m_features = len(self.X[0])
        self.w = np.zeros(self.m_features + 1)
        self.cost = []
        self.iteration = []

    @staticmethod
    def __normalize(x, y):
        x_n = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        y_n = (y - np.amin(y)) / (np.amax(y) - np.amin(y))
        return x_n, y_n

    def hypothesis(self, x):
        return x.dot(self.w)

    def compute_cost(self):
        cost = (1 / (2 * self.m_data)) * np.sum(np.square(self.hypothesis(self.X_n) - self.y_n))
        return cost

    def fit_model(self, epoch=1000000, lr=0.01):
        # Normalize data
        self.X_n, self.y_n = self.__normalize(self.X, self.y)
        # Add a column of 1 for the hypothesis
        col = np.ones((self.m_data, 1))
        self.X_n = np.hstack((self.X_n[:, :0], col, self.X_n[:, 0:]))

        for i in range(epoch):
            self.w = self.w - (lr / self.m_data) * (self.hypothesis(self.X_n) - self.y_n).T.dot(self.X_n)
            self.cost.append(self.compute_cost())
            self.iteration.append(i)

    def predict(self, x, y):
        x_n, y_n = self.__normalize(x, y)
        col = np.ones((len(y), 1))
        x_n = np.array(x_n)
        x_n = np.hstack((x_n[:, :0], col, x_n[:, 0:]))
        y_ = self.hypothesis(x_n)
        y_ = y_ * (np.amax(y) - np.amin(y)) + np.amin(y)
        for real, prediction in zip(y, y_):
            print(f'Real value: {real} -> predicted value: {prediction}')

        plt.plot(self.iteration, self.cost)
        plt.show()
