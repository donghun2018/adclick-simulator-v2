# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:16:33 2017

@author: Piotr Zioło
"""

import numpy as np


def dict_apply(d, f):
    return { key: f(value) for key, value in d.items()}


def add_dicts(dict_1, dict_2):
    return {key: dict_1.get(key, 0) + dict_2.get(key, 0) for key in dict_1}


class LessPrecise(float):
    def __repr__(self):
        return str(self)


def format_dict(d):
    for key, value in d.items():
        if type(value) in [float, np.float64]:
            d[key] = LessPrecise(np.round(value, 4))
        elif type(value) in [list, np.array]:
            d[key] = [LessPrecise(np.round(v, 4)) for v in value]
        elif type(value) in [dict]:
            d[key] = format_dict(d[key])
    return d


if __name__ == "__main__":
    import random

    dim1 = 3
    dim2 = 2
    dim3 = 2

    d = {}
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                d[(i, j, k)] = random.randint(1, 10)


    def f(x):
        return x**2


    print(d)
    print(dict_apply(d, f))