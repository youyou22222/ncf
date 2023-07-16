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
    def __init__(self, n_user, n_items, n_factors,
                 layer_size, name="neumf",
                 gmf_trainable=True, mlp_trainable=True,
                 gmf_pretrain=None, mlp_pretrain=None):
        """
        args
        param n_user: Number of users in Dataset
        param n_items: Number of items in Dataset
        param n_factors: Dimension of the latent embedding vectors.
        param layer_size: List of layer sizes for MLP.
        param name: Name of the NCF model.
        param gmf_trainable: Whether the gmf embedding is trainable.
        param mlp_trainable: Whether the mlp embedding is trainable.
        param gmf_pretrain: Pretrain weights for gmf embedding,  model.load_weights.
        param mlp_pretrain: Pretrain weights for mlp embedding, model.load_weights.
        """
        super(Ncf, self).__init__(name=name)
        self.n_user = n_user
        self.n_items = n_items
        self.n_factors = n_factors
        self.layer_size = layer_size
        self.model_name = name
        self.gmf_trainable = gmf_trainable
        self.mlp_trainable = mlp_trainable
        self.gmf_pretrain = gmf_pretrain
        self.mlp_pretrain = mlp_pretrain

        model_options = ["gmf", "mlp", "neumf"]
        if self.model_name not in model_options:
            raise ValueError(
                "The argument model is invalid, please input one of the following options: gmf, mlp, neumf")

        if name == "gmf":
            self.gmf = GMFLayer(self.n_user, self.n_items, self.n_factors, trainable=self.gmf_trainable)
            if self.gmf_pretrain:
                self.gmf.set_weights(self.gmf_pretrain)
        if name == "mlp":
            self.mlp = MlpLayer(self.n_user, self.n_items, self.n_factors, self.layer_size, trainable=self.mlp_trainable)
            if self.mlp_pretrain:
                self.mlp.set_weights(self.mlp_pretrain)
        if name == "neumf":
            self.gmf = GMFLayer(self.n_user, self.n_items, self.n_factors, trainable=self.gmf_trainable)
            self.mlp = MlpLayer(self.n_user, self.n_items, self.n_factors//2, self.layer_size, trainable=self.mlp_trainable)
            if self.gmf_pretrain:
                self.gmf.set_weights(self.gmf_pretrain)
            if self.mlp_pretrain:
                self.mlp.set_weights(self.mlp_pretrain)



    def build(self, input_shape):
        """
        args
        param input_shape: Shape of the input tensor.
        """
        super(Ncf, self).build(input_shape)
        print("input_shape: ", input_shape)


    def call(self, inputs):
        """
        args
        param inputs: Input tensor of shape (batch_size, 2).
        """
        if self.model_name == "gmf":
            output = self.gmf(inputs)
        elif self.model_name == "mlp":
            output = self.mlp(inputs)
        elif self.model_name == "neumf":
            concat = tf.concat([self.gmf(inputs), self.mlp(inputs)], axis=1)
            output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)
        return tf.sigmoid(output)


if __name__ == "__main__":
    ncf = Ncf(100, 100, 10, [10, 10], "neumf")
    r = ncf(tf.constant([[1, 2], [3, 4]]))
    print(r)

