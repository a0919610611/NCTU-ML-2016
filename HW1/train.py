#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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
    # rs
    clf_rs = RandomForestClassifier(n_estimators=19,
                                    criterion='gini',
                                    min_samples_split=4,
                                    random_state=37,
                                    max_features=None,
                                    warm_start=True,
                                    max_depth=None,
                                    )
    clf_rs.fit(X, Y)
    predicted_rs = clf_rs.predict(X)
    cfm_rs = confusion_matrix(Y, predicted_rs, labels=[
        'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

    # KFold
    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)
    cfm_KFold = np.zeros(shape=(3, 3,), dtype=np.int64)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf_KFold = RandomForestClassifier(n_estimators=19,
                                           criterion='gini',
                                           min_samples_split=4,
                                           random_state=37,
                                           max_features=None,
                                           warm_start=True,
                                           max_depth=None,
                                           )
        clf_KFold.fit(X_train, Y_train)
        predicted_KFold = clf_KFold.predict(X_test)
        cfm_ThisFold = confusion_matrix(Y_test,
                                        predicted_KFold,
                                        labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
                                        )
        cfm_KFold += cfm_ThisFold
    print('Resubstitution is\n')
    plt.figure()
    plot_confusion_matrix(cfm_rs,
                          classes=target_names,
                          title='resubstition')
    print('KFold is\n')
    plt.figure()
    plot_confusion_matrix(cfm_KFold,
                          classes=target_names,
                          title='KFold (k=%d)' % k)
    plt.show()
