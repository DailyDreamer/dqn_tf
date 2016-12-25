import tensorflow as tf

def clipped_error(x):
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv2d(x, y_dim, kernel_size, stride, name='conv2d'):
  with tf.variable_scope(name):
    stride = [1, 1, stride[0], stride[1]]
    kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape().as_list()[1], y_dim]

    w = tf.get_variable('w', kernel_shape, tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', [y_dim], 
        initializer=tf.constant_initializer(0.1))

    conv = tf.nn.conv2d(x, w, stride, 'VALID', data_format='NCHW')
    y = tf.nn.relu(tf.nn.bias_add(conv, b, data_format))
  return y, w, b

def linear(x, y_dim, activation_fn=tf.nn.relu, name='linear'):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [x.get_shape().as_list()[1], y_dim], tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', [y_dim],
        initializer=tf.constant_initializer(0.1))

    y = tf.nn.bias_add(tf.matmul(x, w), b)

    if activation_fn != None:
      return activation_fn(y), w, b
    else:
      return y, w, b