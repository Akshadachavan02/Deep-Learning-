import tensorflow as tf

class OurDenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_output_nodes):
        super(OurDenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes

    def build(self, input_shape):
        d = int(input_shape[-1])
        # Define and initialize parameters: weight matrix W and bias b
        self.W = self.add_weight(name="weight", shape=[d, self.n_output_nodes])  # Shape: (input_dim, output_dim)
        self.b = self.add_weight(name="bias", shape=[1, self.n_output_nodes])    # Shape: (1, output_dim)

    def call(self, x):
        # Compute z = xW + b
        z = tf.matmul(x, self.W) + self.b

        # Apply sigmoid activation: y = sigmoid(z)
        y = tf.sigmoid(z)
        return y

# Since layer parameters are initialized randomly, we will set a random seed for reproducibility
tf.keras.utils.set_random_seed(1)

# Create an instance of OurDenseLayer with 3 output nodes
layer = OurDenseLayer(3)

# Build the layer with an input shape of (1, 2)
layer.build((1, 2))

# Define the input tensor
x_input = tf.constant([[1, 2.]], shape=(1, 2))

# Call the layer on the input to compute the output
y = layer.call(x_input)

# Test the output
print("Output of the layer:", y.numpy())
