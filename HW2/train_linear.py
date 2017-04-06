#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from parse_data import get_data
from draw_confusion_matrix import plot_confusion_matrix
import time
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import KFold


if __name__ == '__main__':
    data = get_data()
    feature_matrix = data['feature_matrix']
    target_vector = data['target_vector']
    KNN_rs = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
    start_time = time.time()
    KNN_rs.fit(feature_matrix, target_vector)
    used_time = time.time() - start_time
    print("Linear Search training time is %s seconds" % (used_time))
    start_time = time.time()
    predicted = KNN_rs.predict(feature_matrix)
    used_time = time.time() - start_time
    print("Linear Search querying time is %s seconds" % (used_time))
