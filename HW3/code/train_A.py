import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    df = pd.read_csv('train_data_a.csv')
    X = np.matrix(df.iloc[:, :-1])
    Y = np.array(df.iloc[:, -1])
    GNB = GaussianNB()
    GNB.fit(X, Y)
    Test_X = np.matrix([[222, 4.5, 1518, 74, 0.25, 1642]])
    print(GNB.predict(Test_X))
