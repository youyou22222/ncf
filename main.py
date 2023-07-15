# coding:utf-8

import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
from layers import LossLayer
from utils import get_optimizer_instance, get_metric_instance, train
from ncf import Ncf
from dataset import Dataset

# unit test
if __name__ == "__main__":
    # test get_optimizier_instance

    optimizer = get_optimizer_instance(optimizer_name="adam", learning_rate=0.001)
    loss_fn = LossLayer("logloss")
    metric = get_metric_instance("accuracy")
    data = Dataset("./data/ml-1m")
    ncf_model = Ncf(n_user=data.num_users, n_items=data.num_items, n_factors=8, layer_size=[64, 32, 16, 8])
    train_data, test_ratings, negative_ratings = data.trainMatrix, data.testRatings, data.testNegatives
    train(train=train_data,
          test=None, model=ncf_model,
          optimizer=optimizer,
          loss_fn=loss_fn, metric=metric, epoches=10)
