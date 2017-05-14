#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np


def clean_data(filename):
    class1 = set()
    class2 = set()
    class3 = set()
    class4 = set()
    class5 = set()
    class6 = set()
    with open("class1_out.txt", "r") as fin:
        In = fin.read()
        In = In.split(',')
        for i in In:
            class1.add(i)
    with open("class2_out.txt", "r") as fin:
        In = fin.read()
        In = In.split(',')
        for i in In:
            class2.add(i)
    with open("class3_out.txt", "r") as fin:
        In = fin.read()
        In = In.split(',')
        for i in In:
            class3.add(i)
    with open("class4_out.txt", "r") as fin:
        In = fin.read()
        In = In.split(',')
        for i in In:
            class4.add(i)
    with open("class5_out.txt", "r") as fin:
        In = fin.read()
        In = In.split(',')
        for i in In:
            class5.add(i)
    with open("class6_out.txt", "r") as fin:
        In = fin.read()
        In = In.split(',')
        for i in In:
            class6.add(i)
    df = pd.read_csv(filename, header=None)
    fa = filename.split('.')
    bad_index = []
    for idx, i in enumerate(df.values):
        if '?' in i:
            bad_index.append(idx)
    df.drop(df.index[bad_index], inplace=True)
    feature_matrix = np.matrix(df[df.columns[1:]].values)
    date = df[df.columns[0]]
    target_values = []
    for i in date:
        if i in class1:
            target_values.append('class1')
        elif i in class2:
            target_values.append('class2')
        elif i in class3:
            target_values.append('class3')
        elif i in class4:
            target_values.append('class4')
        elif i in class5:
            target_values.append('class5')
        elif i in class6:
            target_values.append('class6')
    # print(target_values)

    target_values = pd.Series(target_values)
    target_values.index = df.index
    df = df[df.columns[1:]]
    df = pd.concat([df, target_values], axis=1)
    df.to_csv(fa[0] + "_clean." + fa[1], mode='w', index=False, header=None)


def get_data(filename):
    df = pd.read_csv(filename, header=None)
    data = dict()
    data['feature_matrix'] = np.matrix(df[df.columns[0:-1]].values)
    data['target_vector'] = df[df.columns[-1]].values.ravel()
    return data


if __name__ == "__main__":
    clean_data(sys.argv[1])
