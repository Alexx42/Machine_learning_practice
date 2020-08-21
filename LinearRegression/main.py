from LinearRegression import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv('dataset/data.csv')

    label = 'Y house price of unit area'
    features = ['X2 house age', 'X3 distance to the nearest MRT station',
                'X4 number of convenience stores', 'X5 latitude']

    partition = int(len(df) * 0.95)
    X_train, X_test = df[features].iloc[:partition], df[features].iloc[partition:]
    y_train, y_test = df[label].iloc[:partition], df[label].iloc[partition:]

    regression = LinearRegression(X_train, y_train)
    regression.fit_model()

    regression.predict(X_test, y_test)
