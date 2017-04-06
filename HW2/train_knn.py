#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from parse_data import get_data
from draw_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
import time
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, DistanceMetric
from sklearn.model_selection import KFold

if __name__ == '__main__':
    data = get_data()
    feature_matrix = data['feature_matrix']
    target_vector = data['target_vector']
    KNN_rs = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', metric='euclidean')
    start_time = time.time()
    KNN_rs.fit(feature_matrix, target_vector)
    used_time = time.time() - start_time
    print("KD Tree using euclidean distance with resubstitution method training time is %s seconds" % used_time)
    start_time = time.time()
    predicted_rs = KNN_rs.predict(feature_matrix)
    used_time = time.time() - start_time
    print("KD Tree using euclidean distance with resubstitution method querying time is %s seconds" % used_time)
    cfm_rs = confusion_matrix(target_vector, predicted_rs, labels=range(1, 11))
    print("KD Tree using euclidean distance with resubstitution method confusion matrix is ")
    print(cfm_rs)
    ac_rate = accuracy_score(target_vector, predicted_rs)
    print("KD Tree using euclidean distance with resubstitution method accurary score is %s" % ac_rate)
    plt.figure()
    plot_confusion_matrix(cfm_rs, classes=range(1, 11),
                          title='KD Tree using euclidean distance with resubstitution method')
    plt.show()
