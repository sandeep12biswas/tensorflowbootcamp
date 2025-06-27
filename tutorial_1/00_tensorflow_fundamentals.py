import tensorflow as tf
from numpy.matrixlib.defmatrix import matrix

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
print(vector)

print(vector.ndim)

matrix = tf.constant([[7,10],
                     [10,7]])
print(matrix)
print(matrix.ndim)

another_matrix = tf.constant([[10.,7.],
                              [11.,5.],
                              [5.,5.]], dtype=tf.float16)
print(another_matrix)
print(another_matrix.ndim)

# changeable and unachaneable variables
changeable_tensor = tf.Variable([10., 7.], dtype=tf.float16)
unchaneable_tenson = tf.constant([10.,7.], dtype=tf.float16)
print(changeable_tensor)
print(unchaneable_tenson)