#coding:utf-8
"""
train ncf model
"""

import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adadelta


def get_optimizier_instance(optimizer_name, learning_rate):
    """
    args
    param learning_rate: Learning rate of optimizer.
    """
    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == "adagrad":
        optimizer = Adagrad(learning_rate=learning_rate)
    elif optimizer_name == "adadelta":
        optimizer = Adadelta(learning_rate=learning_rate)
    else:
        raise ValueError("optimizer_name must be adam, sgd, rmsprop, adagrad or adadelta")
    return optimizer

def get_metric_instance(metric_name):
    """
    args
    param metric_name: Name of metric.
    """
    if metric_name == "auc":
        metric = tf.keras.metrics.AUC()
    elif metric_name == "accuracy":
        metric = tf.keras.metrics.Accuracy()
    elif metric_name == "precision":
        metric = tf.keras.metrics.Precision()
    elif metric_name == "recall":
        metric = tf.keras.metrics.Recall()
    elif metric_name == "mae":
        metric = tf.keras.metrics.MeanAbsoluteError()
    elif metric_name == "mse":
        metric = tf.keras.metrics.MeanSquaredError()
    else:
        raise ValueError("metric_name must be auc, accuracy, precision, recall, mae or mse")
    return metric
def train(model, train, test, valid, metric, optimizer, loss_fn, epoches):
    for epoch in range(epoches):
        start_time = time.time()
        train_loss = []
        for (batch, (user, item, label)) in enumerate(train):
            with tf.GradientTape() as tape:
                logits = model([user, item])
                loss = loss_fn(label, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            metric.update_state(label, logits)
            train_loss.append(loss)
            if batch % 100 == 0:
                print("epoch: {}, batch: {}, loss: {}, accuracy: {}".format(epoch, batch, sum(train_loss)/len(train_loss), metric.result()))
        print("epoch: {}, loss: {}, accuracy: {}".format(epoch, sum(train_loss)/len(train_loss), metric.result()))
        metric.reset_states()


