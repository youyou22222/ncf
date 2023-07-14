#coding:utf-8

import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
from layers import LossLayer
from utils import get_optimizier_instance, get_metric_instance, train
from ncf import Ncf

# unit test
if __name__ == "__main__":
    # test get_optimizier_instance
    optimizer = get_optimizier_instance("adam", 0.01)
    print(optimizer)
    optimizer = get_optimizier_instance("sgd", 0.01)
    print(optimizer)
    optimizer = get_optimizier_instance("rmsprop", 0.01)
    print(optimizer)
    optimizer = get_optimizier_instance("adagrad", 0.01)
    print(optimizer)
    optimizer = get_optimizier_instance("adadelta", 0.01)
    print(optimizer)
    # test get_metric_instance
    metric = get_metric_instance("auc")
    print(metric)
    metric = get_metric_instance("accuracy")
    print(metric)
    metric = get_metric_instance("precision")
    print(metric)
    metric = get_metric_instance("recall")
    print(metric)
    metric = get_metric_instance("mae")
    print(metric)
    metric = get_metric_instance("mse")
    print(metric)
    # test train
    model = Ncf(10, 10, 10, [16,8,4])
    train(model, None, None, None, None, None, None, 10)
    print("unit test pass")