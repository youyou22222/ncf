#coding:utf7
"""
Neural Collaborative Filtering (NCF)
# Reference:
- Xiangnan He et al. Neural Collaborative Filtering. In WWW 2017.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from layers import GMFLayer, MlpLayer

class Ncf(Model):
    def __init__(self, n_user, n_items, n_factors, layer_size, name="ncf"):
        """
        args
        param n_user: Number of users in Dataset
        param n_items: Number of items in Dataset
        param n_factors: Dimension of the latent embedding vectors.
        param layer_size: List of layer sizes for MLP.
        param name: Name of the NCF model.
        """
        super(Ncf, self).__init__(name=name)
        self.n_user = n_user
        self.n_items = n_items
        self.n_factors = n_factors
        self.layer_size = layer_size
        self.model_name = name

    def call(self, inputs):
        """
        args
        param inputs: Input tensor of shape (batch_size, 2).
        """

        model_options = ["gmf", "mlp", "neumf"]
        if self.model_name not in model_options:
            raise ValueError("The argument model is invalid, please input one of the following options: gmf, mlp, neumf")

        if self.model_name == "gmf":
            output = self.gmf(inputs)
        elif self.model_name == "mlp":
            output = self.mlp(inputs)
        elif self.model_name == "neumf":
            gmf = self.gmf(inputs)
            mlp = self.mlp(inputs)
            concat = tf.concat([gmf, mlp], axis=1)
            output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)




        gmf = GMFLayer(self.n_user, self.n_items, self.n_factors)
        mlp = MlpLayer(self.n_user, self.n_items, self.n_factors, self.layer_size)
        gmf_out = gmf(inputs)
        mlp_out = mlp(inputs)
        concat = tf.concat([gmf_out, mlp_out], axis=1)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)
        return output