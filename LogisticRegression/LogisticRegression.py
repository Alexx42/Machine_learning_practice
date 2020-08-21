import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.m_data = self.x.shape[0]
        self.n_features = self.x.shape[1]
        self.w = np.zeros((self.n_features + 1, 1))
        self.cost = []
        self.iteration = []

    @staticmethod
    def __normalize(data):
        return (data - np.amin(data)) / (np.amax(data) - np.amin(data))

    def prepare_data(self):
        self.x, self.y = self.__normalize(self.x), self.__normalize(self.y)

        col = np.ones((self.m_data, 1))
        self.x = np.hstack((self.x[:, :0], col, self.x[:, 0:]))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def hypothesis(self):
        return self.x.dot(self.w)

    def compute_cost(self):
        return sum(-self.y * (np.log10(self.sigmoid(self.hypothesis()))) -\
               (1 - self.y) * (np.log10(1 - self.sigmoid(self.hypothesis()))))

    def fit_model(self, epoch=5000000, lr=1):
        for i in range(epoch):
            self.w = self.w - (lr / self.m_data) * self.x.T.dot(self.sigmoid(self.hypothesis()) - self.y)
            self.iteration.append(i)
            self.cost.append((1 / self.m_data) * self.compute_cost())
        plt.plot(self.iteration, self.cost)
        plt.show()

    def predict(self, x, y):
        x = np.array(x)
        y = np.array(y)

        x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))

        col = np.ones((x.shape[0], 1))
        x = np.hstack((x[:, :0], col, x[:, 0:]))

        y_predict = self.sigmoid(x.dot(self.w))
        right = 0
        for r, v in zip(y, y_predict):
            t = v
            if v > 0.5:
                v = 1
            else:
                v = 0
            if v == r:
                right += 1
            print(f'The real value is: {r} -> {v} {t}')
        print(f'Precision -> {right / y.shape[0]}')