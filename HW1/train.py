#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

k = 5

with open('./iris.data', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    feature_names = np.array(['sl', 'sw', 'pl', 'pw', 'class'])
    X, Y = [], []
    for row in reader:
        X.append(row[:4])
        Y.append(row[4])
    X = np.array(X)
    Y = np.array(Y)
    # rs
    clf_rs = RandomForestClassifier(min_samples_split=3, class_weight='balanced', random_state=37, max_features=None)
    clf_rs.fit(X, Y)
    predicted_rs = clf_rs.predict(X)
    cfm_rs = confusion_matrix(Y, predicted_rs, labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

    # KFold
    # clf_k_fold = RandomForestClassifier(min_samples_split=3, class_weight='balanced', random_state=23)
    kf = KFold(n_splits=k, random_state=True)
    kf.get_n_splits(X)
    cfm_KFold = np.zeros(shape=(3, 3,), dtype=np.int64)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf_KFold = RandomForestClassifier(min_samples_split=3,
                                           class_weight='balanced',
                                           random_state=23,
                                           max_features=None
                                           )
        clf_KFold.fit(X_train, Y_train)
        predicted_KFold = clf_KFold.predict(X_test)
        cfm_ThisFold = confusion_matrix(Y_test,
                                        predicted_KFold,
                                        labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
                                        )
        print(cfm_ThisFold)
        cfm_KFold += cfm_ThisFold
    print('Resubstitution is', cfm_rs)
    print('KFold is', cfm_KFold)
