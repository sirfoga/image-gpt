import os
import json
import time
import pickle
import subprocess
import math

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def iter_data(*datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf")):
    n = datas[0].shape[0]
    if truncate:
        n = (n//n_batch)*n_batch
    n = min(n, max_batches*n_batch)
    n_batches = 0
    for i in range(0, n, n_batch):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i+n_batch]
        else:
            yield (d[i:i+n_batch] for d in datas)
        n_batches += 1

def squared_euclidean_distance(a, b):
    b = tf.transpose(b)
    a2 = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
    b2 = tf.reduce_sum(tf.square(b), axis=0, keepdims=True)
    ab = tf.matmul(a, b)
    d = a2 - 2*ab + b2
    return d

def color_quantize(x, np_clusters):
    clusters = tf.Variable(np_clusters, dtype=tf.float32, trainable=False)
    x = tf.reshape(x, [-1, 3])
    d = squared_euclidean_distance(x, clusters)
    return tf.argmin(d, 1)

def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def stuff2pickle(stuff, f_path):
    with open(f_path, 'wb') as fp:
        pickle.dump(stuff, fp)

def load_pickle(f_path):
    with open(f_path, 'rb') as fp:
        return pickle.load(fp)
