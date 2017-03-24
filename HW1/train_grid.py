#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

k = 10

with open('./iris.data', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    feature_names = np.array(['sl', 'sw', 'pl', 'pw', 'class'])
    target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    X, Y = [], []
    for row in reader:
        X.append(row[:4])
        Y.append(row[4])
    X = np.array(X)
    Y = np.array(Y)
    parameters = {
        'n_estimators': range(10, 20),
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', None],
        'min_samples_split': range(2, 5),
        'warm_start': [True, False],
        'max_depth': [None, 1, 2, 3, 4, 5]
    }
    rf = RandomForestClassifier(random_state=37, n_estimators=10)
    clf = GridSearchCV(rf, parameters, cv=k, n_jobs=5)
    clf.fit(X, Y)
    print(clf.best_score_)
    print(clf.best_params_)
