import tensorflow as tf
import numpy as np
import display_tensor_data as dd

# check the tensor version

version = tf.__version__
print(version)

# create tensor with tf.constant()
scaler = tf.constant(7)
print(scaler)

#Check the number of dimensions of tenson (ndim stands for number of dimensions)

print(scaler.ndim)

#vectos constants

vector = tf.constant([10,10])
print("vector value is {}".format(vector))

print("vector.ndim is {}".format(vector.ndim))

matrix = tf.constant([[7,10],
                     [10,7]])
print("matrix value is {}".format(matrix))
print(matrix.ndim)

another_matrix = tf.constant([[10.,7.],
                              [11.,5.],
                              [5.,5.]], dtype=tf.float16)
print("another_matrix value is {}".format(another_matrix))
print("another_matrix.ndim value is {}".format(another_matrix.ndim))

# changeable and unreachable variables
changeable_tensor = tf.Variable([10., 7.], dtype=tf.float16)
unchangeable_tensor = tf.constant([10., 7.], dtype=tf.float16)
print(changeable_tensor)
print(unchangeable_tensor)

print('changeable variable-> {}'.format(changeable_tensor[0]))
changeable_tensor[0].assign(17)
print('changeable variable after change->',changeable_tensor)

# Creating random tensors

random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3,2))
print("random_1 value is {}".format(random_1))

## NUmpy operations
numpy_A = np.arange(1, 25, dtype=np.int32)
print("numpy_array {}".format(numpy_A))

A = tf.constant(numpy_A, shape=(4,3,2))
#print("A value is {}".format(A))

## DDifferent tensor elements

numpy_gen=np.arange(1, 121, dtype=np.int32)
rank_4_tenson=tf.constant(numpy_gen, shape=(2,3,4,5))
dd.display_tensor(rank_4_tenson)

print(rank_4_tenson[:2, :2, :2, :2])