#!/usr/bin/env python3
import numpy as np
import parse_data as ps
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':

    train_data = ps.get_data('train_clean.csv')
    test_data = ps.get_test_matrix('test_clean.csv')

    X = train_data['feature_matrix']
    Y = train_data['target_vector']
    label = test_data['label']
    TestX = test_data['feature_matrix']
    GNB = GaussianNB()
    GNB.fit(X, Y)
    Predict_Y = GNB.predict(X)
    TestY = GNB.predict(TestX)
    for i in range(len(label)):
        print("%s is %s" % (label[i][0], TestY[i]))
