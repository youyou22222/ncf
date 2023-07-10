import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding



class MlpLayer(Layer):
    def __init__(self, n_user, n_items, n_factors, layer_size, name="mlp_layer"):
        """
        args
        param n_user: Number of users in Dataset
        param n_items: Number of items in Dataset
        param n_factors: Dimension of the latent embedding vectors.
        param layer_size: List of layer sizes for MLP.
        param name: Name of the MLP layer.
        """
        super(MlpLayer, self).__init__(name=name)
        self.layer_size = layer_size
        self.embedding_mlp_user = Embedding(input_dim=n_user, output_dim=n_factors, name="embedding_mlp_user")
        self.embedding_mlp_item = Embedding(input_dim=n_items, output_dim=n_factors, name="embedding_mlp_item")

    def build(self, input_shape):
        """
        args
        param input_shape: Shape of the input tensor.
        """
        super(MlpLayer, self).build(input_shape)
        print("input_shape: ", input_shape)

    def call(self, inputs):
        """
        args
        param inputs: Input tensor of shape (batch_size, 2).
        """
        user_input, item_input = inputs[:, 0], inputs[:, 1]
        user_latent = self.embedding_mlp_user(user_input)
        item_latent = self.embedding_mlp_item(item_input)
        layer = tf.concat([user_latent, item_latent], axis=1)
        for dim in self.layer_size:
            layer = tf.keras.layers.Dense(dim, activation='relu')(layer)
        return layer


if __name__ == "__main__":
    mlp = MlpLayer(100, 100, 10, [10, 10])
    r = mlp(tf.constant([[1, 2], [3, 4]]))
    print(r)