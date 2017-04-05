#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd


def get_data():
    df = pd.read_csv('./winequality-white.csv', sep=';')
    return df


if __name__ == '__main__':
    df = get_data()
    col_names = list(df)
    feature_names = col_names[:-1]
    target_name = col_names[-1]
    # print(feature_names)
    # print(target_name)
    # print(df.values)
    values = df.values
    feature_matrix = df[df.columns[0:-1]].values
    target_vector = df[[-1]].values
    print(feature_matrix)
    print(target_vector)
