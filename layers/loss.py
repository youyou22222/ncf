#coding:utf-8
"""
loss layer define the loss function when training the model
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.losses import mean_squared_error

class LossLayer(Layer):
    def __init__(self, loss_type, name="loss_layer"):
        """
        args
        param loss_type: Type of loss function.
        param name: Name of the loss layer.
        """
        super(LossLayer, self).__init__(name=name)
        self.loss_type = loss_type

    def __call__(self, inputs, outputs):
        """
        args
        param inputs: labels tensor of shape (batch_size, 1).
        param outputs: Output tensor of shape (batch_size, 1).
        """
        if self.loss_type == "logloss":
            # explicit feedback
            loss = binary_crossentropy(inputs, outputs, from_logits=False)
        elif self.loss_type == "mse":
            # implicit feedback
            #loss = mean(square(y_true - y_pred))
            loss = mean_squared_error(inputs, outputs, from_logits=False)
        else:
            raise ValueError("loss_type must be logloss or mse")

        return tf.reduce_mean(loss)


if __name__ == "__main__":
    loss_fn = LossLayer("mse")
    r = loss_fn(tf.constant([[1], [0]]), tf.constant([[0.9], [0.1]]))
    print(r)
