import tensorflow as tf
import numpy as np
import os

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

class Agent(object):
  def __init__(self, conf):
    self.env = Env(conf.env_name, conf.screen_height, conf.screen_width, conf.no_op_max, conf.is_display)
    self.history = History(conf.agent_history_length, conf.screen_height, conf.screen_width)
    self.replay_memory = ReplayMemory(conf.replay_memory_size, conf.screen_height, conf.screen_width, conf.agent_history_length, conf.batch_size)
    self.conf = conf

    # build network
    self.q = QNetwork(conf.agent_history_length, 
      conf.screen_height, 
      conf.screen_width, 
      self.env.action_size, 
      conf.learning_rate, 
      'q'
    )
    self.q_target = QNetwork(
      conf.agent_history_length, 
      conf.screen_height, 
      conf.screen_width, 
      self.env.action_size, 
      conf.learning_rate, 
      'q_target'
    )
    with tf.variable_scope('reward_summary'):
      self.reward_placeholder = tf.placeholder(tf.int32, None, name='episode_reward_summary')
      self.episode_reward_summary_op = tf.summary.scalar(
        self.reward_placeholder.op.name, 
        self.reward_placeholder
      )
    with tf.variable_scope('update_target'):
      self.update_target_op = create_update_target_op(self.q_target, self.q)
    with tf.variable_scope('init'):
      self.init_op = tf.global_variables_initializer()
    
    self.saver = tf.train.Saver()

  def train(self):
    conf = self.conf
    self.summary_writer = tf.summary.FileWriter(conf.summary_dir)
    # init some local variables
    ep = conf.init_ep
    ep_step_drop = (conf.init_ep - conf.final_ep) / conf.final_ep_frame
    current_reward = 0
    best_reward = 0

    screen = self.env.new_random_game()
    for _ in range(conf.agent_history_length):
      self.history.add(screen)
    
    # start training
    with tf.Session() as sess:
      if conf.is_restore_model:
        print('Restore model...')
        ckpt = tf.train.get_checkpoint_state(conf.restore_dir)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')
        basestep = conf.basestep
        ep -= ep_step_drop*basestep
      else:
        sess.run(self.init_op)
        sess.run(self.update_target_op)
        basestep = 0
      for step in range(conf.max_steps):
        # ep-greedy action select
        if np.random.rand(1) < ep:
          action = self.env.random_action
        else: #feed one state, not a batch
          action = sess.run(self.q.actions, feed_dict={self.q.screen_placeholder: [self.history.get()]})[0]
        if ep > conf.final_ep:
          ep -= ep_step_drop
        # take action
        screen, reward, done = self.env.act(action, is_training=True)
        current_reward += reward
        # add to memory
        norm_reward = max(conf.min_reward, min(conf.max_reward, reward))
        self.history.add(screen)
        self.replay_memory.add(screen, action, norm_reward, done)
        # update
        if (step+basestep) % conf.update_freq == 0:
          self.update_q(sess, conf.batch_size, conf.discount, step+basestep)
        if (step+basestep) % conf.target_update_freq == (conf.target_update_freq - 1):
          sess.run(self.update_target_op)

        # game over? start a new episode
        if done:
          self.env.new_random_game()
          if current_reward > best_reward:
            best_reward = current_reward
          print("progress %f, episode reward %d, best reward %d " % (step/conf.max_steps, current_reward, best_reward))
          episode_reward_summary = sess.run(self.episode_reward_summary_op, feed_dict={
            self.reward_placeholder:current_reward
          })
          self.summary_writer.add_summary(episode_reward_summary, step+basestep)
          current_reward = 0

        if (step+basestep) % conf.save_freq == (conf.save_freq - 1):
          save_path = conf.model_dir+str(step+basestep)+'.ckpt'
          self.saver.save(sess, save_path, step+basestep)
      
    self.summary_writer.close()


  def update_q(self, sess, batch_size, discount, step):
    if self.replay_memory.count < batch_size:
      return
    s, a, r, next_s, done = self.replay_memory.sample()

    next_q_max = sess.run(self.q_target.q_max, feed_dict={self.q_target.screen_placeholder: next_s})
    q_target_value = r + (1.0 - done) * discount * next_q_max

    _, loss_summary = sess.run([self.q.optimizer, self.q.loss_summary_op], feed_dict={
      self.q.screen_placeholder: s,
      self.q.q_target_placeholder: q_target_value,
      self.q.action_placeholder: a,
    })
    self.summary_writer.add_summary(loss_summary, step)
  
  def play(self, n_step=10000, n_episode=100, test_ep=0.1):
    conf = self.conf

    print('Restore model...')
    ckpt = tf.train.get_checkpoint_state(conf.restore_dir)
    self.saver.restore(sess, ckpt.model_checkpoint_path)
    print('Model restored')

    if not conf.is_display:
      self.env.env.monitor.start(conf.gym_dir)

    best_reward = 0
    for i in xrange(n_episode):
      current_reward = 0
      screen = self.env.new_random_game()
      for _ in range(conf.agent_history_length):
        self.history.add(screen)
      for _ in range(n_step):
        # ep-greedy action select
        if np.random.rand(1) < test_ep:
          action = self.env.random_action
        else: #feed one state, not a batch
          action = sess.run(self.q.actions, feed_dict={self.q.screen_placeholder: [self.history.get()]})[0]
        # take action
        screen, reward, done = self.env.act(action, is_training=False)
        # add to memory
        self.history.add(screen)

        current_reward += reward
        if done:
          break
      if current_reward > best_reward:
        best_reward = current_reward
      print("episode %d/%d, reward %d, best reward %d " % (i, n_episode, current_reward, best_reward))

    if not conf.is_display:
      self.env.env.monitor.close()