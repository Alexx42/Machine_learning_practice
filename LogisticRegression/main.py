from LogisticRegression import LogisticRegression
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('heart.csv')

    df = df.sample(frac=1).reset_index(drop=True)
    features = [
        'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    label = [
        'target'
    ]

    partition = int(len(df) * 0.90)

    x_train, y_train, x_test, y_test = df[features].iloc[:partition], df[label].iloc[:partition],\
                                       df[features].iloc[partition:], df[label].iloc[partition:]

    classification = LogisticRegression(x_train, y_train)
    classification.prepare_data()
    classification.fit_model()
    classification.predict(x_test, y_test)

