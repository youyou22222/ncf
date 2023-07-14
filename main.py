# coding:utf-8

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

    # test data for model
    # (user, item, label)

    train_data = [
        [
        tf.constant([[0], [1], [2]]),
        tf.constant([[1], [2], [3]]),
        tf.constant([[1], [0], [1]])
        ]
    ]
    optimizer = get_optimizier_instance(optimizer_name="adam", learning_rate=0.01)
    loss_fn = LossLayer("logloss")
    metric = get_metric_instance("accuracy")
    ncf_model = Ncf(n_user=3, n_items=10, n_factors=8, layer_size=[16, 8, 4])
    train(train=train_data,
          valid=None, test=None, model=ncf_model,
          optimizer=optimizer,
          loss_fn=loss_fn, metric=metric, epoches=10)
    print(tf.__version__)
