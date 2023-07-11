#coding:utf-8
# Author: youyou22222
"""
Generalized Matrix Factorization (GMF) layer
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding

class GMFLayer(Layer):
    def __init__(self, n_user, n_items, n_factors, name="gmf_layer"):
        """
        args
        param n_user: Number of users in Dataset
        param n_items: Number of items in Dataset
        param n_factors: Dimension of the latent embedding vectors.
        param name: Name of the GMF layer.
        """
        super(GMFLayer, self).__init__(name=name)
        self.embedding_gmf_user = Embedding(input_dim=n_user, output_dim=n_factors, name="embedding_gmf_user")
        self.embedding_gmf_item = Embedding(input_dim=n_items, output_dim=n_factors, name="embedding_gmf_item")

    def build(self, input_shape):
        """
        args
        param input_shape: Shape of the input tensor.
        """
        super(GMFLayer, self).build(input_shape)
        print("input_shape: ", input_shape)

    def call(self, inputs):
        """
        args
        param inputs: Input tensor of shape (batch_size, 2).
        """
        user_input, item_input = inputs[:, 0], inputs[:, 1]
        user_latent = self.embedding_gmf_user(user_input)
        item_latent = self.embedding_gmf_item(item_input)
        element_product = tf.multiply(user_latent, item_latent)
        return element_product


if __name__ == "__main__":
    gmf = GMFLayer(100, 100, 10)
    r = gmf(tf.constant([[1, 2], [3, 4]]))
    print(r)
