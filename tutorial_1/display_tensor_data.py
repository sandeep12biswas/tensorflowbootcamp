import tensorflow as tf
# method to print the tensor details taking tensor as method parameter

def display_tensor(tensor):
    print("##################################################")
    print("tensor is {}".format(tensor))
    print("Data type of every element {}".format(tensor.dtype))
    print("Number of dimensions (rank) {}".format(tensor.ndim))
    print("Shape of tenson {}".format(tensor.shape))
    print("Element along the 0 axis {}".format(tensor.shape[0]))
    print("Element along the last axis {}".format(tensor.shape[-1]))
    print("total numner of tenson in element {}".format(tf.size(tensor)))
    print("##################################################")
