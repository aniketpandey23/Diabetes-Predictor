import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

if __name__ == "__main__":
    dataset = pd.read_csv('diabetes.csv')

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    model = LogisticRegression()

    model.fit(x_train, y_train)

    file = open('model.pkl', 'wb')
    pickle.dump(model, file)

    file.close()

