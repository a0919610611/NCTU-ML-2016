#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from parse_data import get_data
from draw_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import time
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, DistanceMetric
from sklearn.model_selection import KFold

if __name__ == '__main__':
    data = get_data()
    X = data['normalized_feature_matrix']
    Y = data['target_vector']
    KNN_rs = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', metric='euclidean')
    start_time = time.time()
    KNN_rs.fit(X, Y)
    used_time = time.time() - start_time
    print("KD Tree using cosine distance with resubstitution method training time is %s seconds" % used_time)
    start_time = time.time()
    predicted_rs = KNN_rs.predict(X)
    used_time = time.time() - start_time
    print("KD Tree using cosine distance with resubstitution method querying time is %s seconds" % used_time)
    cfm_rs = confusion_matrix(Y, predicted_rs, labels=range(1, 11))
    print("KD Tree using cosine distance with resubstitution method confusion matrix is ")
    print(cfm_rs)
    ac_rate = accuracy_score(Y, predicted_rs)
    print("KD Tree using cosine distance with resubstitution method accurary score is %s" % ac_rate)
    plt.figure()
    plot_confusion_matrix(cfm_rs, classes=range(1, 11),
                          title='KD Tree using cosine distance with resubstitution method')

    k = 20
    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)
    cfm_kfold = np.zeros(shape=(10, 10,), dtype=np.int64)
    i = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf_KFold = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', metric='euclidean')
        start_time = time.time()
        clf_KFold.fit(X_train, Y_train)
        used_time = time.time() - start_time
        print("%d fold traing time is %s" % (i, used_time))
        start_time = time.time()
        predicted_thisFold = clf_KFold.predict(X_test)
        used_time = time.time() - start_time
        print("%d fold query time is %s" % (i, used_time))
        cfm_thisFold = confusion_matrix(Y_test, predicted_thisFold, labels=range(1, 11))
        print("%d fold confusion matrix is" % i)
        print(cfm_thisFold)
        print("%d fold ac rate is %s" % (i, accuracy_score(Y_test, predicted_thisFold)))
        cfm_kfold += cfm_thisFold
        i += 1
    plt.figure()
    plot_confusion_matrix(cfm_kfold, classes=range(1, 11),
                          title='KD Tree using cosine distance with KFold method')
    plt.show()
