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
    metric, metric_name = get_metric_instance("auc")
    data = Dataset("./data/ml-1m")
    ncf_model = Ncf(n_user=data.num_users, n_items=data.num_items, n_factors=8, layer_size=[64, 32, 16, 8])
    train_data, test_ratings, negative_ratings = data.trainMatrix, data.testRatings, data.testNegatives
    train(
        model=ncf_model,
        train=train_data,
        test=test_ratings,
        negative_rating=negative_ratings,
        metric=metric, metric_name=metric_name,
        optimizer=optimizer,
        loss_fn=loss_fn, epoches=10, num_negatives=4)
