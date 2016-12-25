import tensorflow as tf
import numpy as np

from env import Env
from history import History
from replay_memory import ReplayMemory
from network import QNetwork

def create_update_target_op(target_network, network):
  copy_ops = []
  for name in target_network.var.keys():
    copy_op = target_network.var[name].assign(network.var[name])
    copy_ops.append(copy_op)
  return tf.group(*copy_ops, name='update_target_op')

class agent(object):
  def __init__(self, conf):
    self.env = Env(conf.env_name, conf.screen_height, conf.screen_width, conf.no_op_max, conf.is_display)
    self.history = History(conf.history_length, conf.screen_height, conf.screen_width)
    self.replay_memory = ReplayMemory(conf.replay_memory_size, conf.screen_height, conf.screen_width, conf.history_length, conf.batch_size)

  def train(self):
    q = QNetwork(conf.agent_history_length, conf.screen_height, conf.screen_width, self.env.action_size, conf.learning_rate, 'q')
    q_target = QNetwork(conf.agent_history_length, conf.screen_height, conf.screen_width, self.env.action_size, conf.learning_rate, 'q_target')
    with tf.variable_scope('init'):
      init_op = tf.initialize_all_variables()
    with tf.variable_scope('update_target'):
      update_target_op = create_update_target_op(q_target, q)

    ep = conf.init_ep
    ep_step_drop = (conf.init_ep - conf.final_ep) / conf.final_ep_frame

    screen = self.env.new_random_game()
    for _ in range(conf.history_length):
      self.history.add(screen)
    
    with tf.Session() as sess:
      sess.run(init_op)
      sess.run(update_target_op)
      for step in range(conf.max_steps):

        # ep-greedy action select
        if np.random.rand(1) < ep:
          action = self.env.random_action
        else: #feed one state, not a batch
          action = sess.run(q.actions, feed_dict={q.screen_placeholder: [self.history.get()]})[0]
        if ep > conf.final_ep:
          ep -= ep_step_drop
        # take action
        screen, reward, done = self.env.act(action, is_training=True)
        # add to memory
        norm_reward = max(conf.min_reward, min(conf.max_reward, reward))
        self.history.add(screen)
        self.replay_memory.add(screen, action, norm_reward, done)
        # update
        if step % conf.update_freq == 0:
          update_q(sess, conf.batch_size, conf.discount)
        if step % conf.target_update_freq == (conf.target_update_freq - 1):
          sess.run(update_target_op)

        # game over? start a new episoid
        if done:
          self.env.new_random_game()
  
  def update_q(self, sess, batch_size, discount):
    if self.replay_memory.count < batch_size:
      return
    s, a, r, next_s, done = self.replay_memory.sample(batch_size)

    next_q_max = sess.run(self.q_target.q_max, feed_dict={self.q_target.screen_placeholder: next_s})
    q_target_value = reward + (1.0 - done) * discount * next_q_max

    _, summary_op = sess.run([self.q.optimizer, self.q.summary], feed_dict={
      self.q.screen_placeholder: s,
      self.q.q_target_placeholder: q_target_value,
      self.q.action_placeholder: a
    })
  
  def play(self):
    pass