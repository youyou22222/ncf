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


def get_optimizer_instance(optimizer_name, learning_rate):
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
        metric = tf.keras.metrics.AUC(num_thresholds=498 )
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
    return metric, metric_name


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users, num_items = train.shape[0], train.shape[1]
    keys = train.keys()
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in keys:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


def train(model, train, test, negative_rating,  metric, metric_name,  optimizer, loss_fn, epoches, num_negatives=4):
    best_metric = 0.0
    train_writer = tf.summary.create_file_writer("./logs")
    for epoch in range(epoches):
        start_time = time.time()
        train_loss = []
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        dataset = tf.data.Dataset.from_tensor_slices(((user_input, item_input), labels))
        dataset = dataset.shuffle(buffer_size=1000).batch(256)

        for batch, data in enumerate(dataset):
            (user, item), label = data
            user = tf.reshape(user, [user.shape[0], 1])
            item = tf.reshape(item, [item.shape[0], 1])
            label = tf.reshape(label, [label.shape[0], 1])
            input, label = tf.concat([user, item], axis=1), tf.cast(label, dtype=tf.float32)
            with tf.GradientTape() as tape:
                logits = model(input)
                loss = loss_fn(label, logits)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if metric_name == "auc":
                label = tf.reshape(label, [-1])
                logits = tf.reshape(logits, [-1])
            metric.update_state(label, logits)
            train_loss.append(loss)
            if batch % 500 == 0:
                print("epoch: {}, batch: {}, loss: {}, {}: {}".format(epoch, batch, sum(train_loss)/len(train_loss), metric_name, metric.result()))
            with train_writer.as_default():
                tf.summary.scalar('batch_loss', loss, step=batch)
                tf.summary.scalar('batch_accuracy', metric.result(), step=batch)
        print("epoch: {}, loss: {}, {}: {}".format(epoch, sum(train_loss)/len(train_loss), metric_name,  metric.result()))
        metric.reset_states()

        cur = evaluate(model, metric, test, negative_rating)
        print("epoch: {}, test {}: {}".format(epoch, metric_name, cur))
        if cur > best_metric:
            best_metric = cur
            if model.model_name in ("gmf", "mlp"):
                if not os.path.exists("./save_weights"):
                     os.makedirs("./save_weights")
                model.save_weights("./save_weights/{}_weights" % model.model_name)
            else:
                checkpoints = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoints.save("./model/{}.ckpt".format(model.model_name))
                model.save_weights("./save_weigts/{}_weights".format(model.model_name))


def evaluate(model, metric, test_ratings, negative_ratings):
    """
    Evaluate the performance
    Return: score of each test rating.
    """
    metric.reset_states()
    test_user_input, test_item_input, test_labels = [], [], []
    for idx in range(len(test_ratings)):
        # positive instance
        u, i = test_ratings[idx]
        test_user_input.append(u)
        test_item_input.append(i)
        test_labels.append(1)
        # negative instances
        neg_samples = negative_ratings[idx]
        for neg_i in neg_samples:
            test_user_input.append(u)
            test_item_input.append(neg_i)
            test_labels.append(0)
    dataset = tf.data.Dataset.from_tensor_slices(((test_user_input, test_item_input), test_labels))
    dataset = dataset.batch(256)
    for batch, data in enumerate(dataset):
        (user, item), label = data
        user = tf.reshape(user, [user.shape[0], 1])
        item = tf.reshape(item, [item.shape[0], 1])
        label = tf.reshape(label, [label.shape[0], 1])
        input, label = tf.concat([user, item], axis=1), tf.cast(label, dtype=tf.float32)
        logits = model(input)
        metric.update_state(label, logits)
    return metric.result()
