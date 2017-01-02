import tensorflow as tf

from op import conv2d, linear, clipped_error

class QNetwork(object):
  def __init__(self, agent_history_length, screen_height, screen_width, action_size, learning_rate, name):
    self.name = name
    self.var = {}

    self.screen_placeholder = tf.placeholder(tf.float32, [None, agent_history_length, screen_height, screen_width], name='screen_placeholder')

    with tf.variable_scope(name):
      self.conv1, self.var['conv1_w'], self.var['conv1_b'] = conv2d(self.screen_placeholder, 32, [8,8], [4,4], 'conv1')
      self.conv2, self.var['conv2_w'], self.var['conv2_b'] = conv2d(self.conv1, 64, [4,4], [2,2], 'conv2')
      self.conv3, self.var['conv3_w'], self.var['conv3_b'] = conv2d(self.conv2, 64, [3,3], [1,1], 'conv3')

      shape = self.conv3.get_shape().as_list()
      self.conv3_flat = tf.reshape(self.conv3, [-1, shape[1]*shape[2]*shape[3]])
      self.fc, self.var['fc_w'], self.var['fc_b'] = linear(self.conv3_flat, 512, name='fc')
      self.q, self.var['q_w'], self.var['q_b'] = linear(self.fc, action_size, activation_fn=None, name='q')

      self.actions = tf.argmax(self.q, 1)
      self.q_max = tf.reduce_max(self.q, 1)

    with tf.variable_scope('optimizer'):
      self.q_target_placeholder = tf.placeholder(tf.float32, [None], name='q_target_placeholder')
      self.action_placeholder = tf.placeholder(tf.int32, [None], name='action_placeholder')

      action_one_hot = tf.one_hot(self.action_placeholder, action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

      self.loss = tf.reduce_mean(clipped_error(self.q_target_placeholder - q_acted), name='loss')
      self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.95, epsilon=0.01).minimize(self.loss)

    with tf.variable_scope('summary'):
      self.q_avg = tf.reduce_mean(self.q_max, name='q_avg')
      q_summary_op = tf.summary.scalar(self.q_avg.op.name, self.q_avg)
      loss_summary_op = tf.summary.scalar(self.loss.op.name, self.loss)
      self.summary_op = tf.merge_summary([q_summary_op, loss_summary_op])
