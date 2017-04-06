#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def get_data():
    df = pd.read_csv('./winequality-white.csv', sep=';')
    col_names = list(df)
    data = dict()
    data['feature_names'] = col_names[:-1]
    data['target_name'] = col_names[-1]
    data['feature_matrix'] = np.matrix(df[df.columns[0:-1]].values)
    data['target_vector'] = df[[-1]].values.ravel()
    return data


if __name__ == '__main__':
    data = get_data()
    for k, v in data.items():
        print(k)
        print(type(v))
