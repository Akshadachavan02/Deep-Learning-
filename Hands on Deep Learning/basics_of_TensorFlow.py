# install tensorflow
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

''' Let's first look at 0-d Tensors, of which a scalar is an example:'''
sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.41421356237, tf.float64)

print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy()))
print("`number` is a {}-d Tensor".format(tf.rank(number).numpy()))

''' Vectors and lists can be used to create 1-d Tensors:'''
sports = tf.constant(["Tennis", "Basketball"], tf.string)
numbers = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)

print("`sports` is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))
print("`numbers` is a {}-d Tensor with shape: {}".format(tf.rank(numbers).numpy(), tf.shape(numbers)))

#TODO difine 2d tensor
sports1 = tf.constant([[1,2,3,4],[5,6,7,8]],tf.int64)

print("sports is a {} -d tensor".format(tf.rank(sports1).numpy()))

'''TODO: Define a 4-d Tensor.'''
# Use tf.zeros to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3.
#   You can think of this as 10 images where each image is RGB 256 x 256.
images = tf.zeros((10,256,256,3),tf.float16)
print(images)

assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"
print("All assertions passed!")

#slicing on tensor 
row_vector = images[1]
column_vector = images[:,1]
scalar = images[0, 1]

print("`row_vector`: {}".format(row_vector.numpy()))
print("`column_vector`: {}".format(column_vector.numpy()))
print("`scalar`: {}".format(scalar.numpy()))

#1.2 Computations on Tensors
c1 = tf.constant(23)
c2 = tf.constant(56)

a = tf.add(c1, c2)
b = c1+c2
print(a)
print(b)

### Defining Tensor computations ###

# Construct a simple computation function
def func(a,b):
    
  c = tf.add(a,b)
  d = tf.subtract(b,1)
  e = tf.multiply(c, d)
  return e

a ,b = 1.5,2.5
e_out = func(a, b)
print(e_out)
















