import tensorflow as tf
from datetime import datetime

from agent import Agent

flags = tf.app.flags
flags.DEFINE_integer('agent_history_length', 4, 'agent_history_length')

flags.DEFINE_integer('screen_height', 84, 'screen_height')
flags.DEFINE_integer('screen_width', 84, 'screen_width')
flags.DEFINE_integer('no_op_max', 30, 'no_op_max')

flags.DEFINE_integer('batch_size', 32, 'batch size')

flags.DEFINE_integer('update_freq', 4, 'Q update freqency')
flags.DEFINE_integer('target_update_freq', 10000, 'target Q update freqency')
flags.DEFINE_integer('save_freq', 10000, 'model save freqency')

flags.DEFINE_float('init_ep', 1.0, 'init_ep')
flags.DEFINE_float('final_ep', 0.1, 'final_ep')
flags.DEFINE_integer('final_ep_frame', 1000000, 'final_ep_frame')

flags.DEFINE_integer('max_reward', 1, 'max_reward')
flags.DEFINE_integer('min_reward', -1, 'min_reward')

flags.DEFINE_float('discount', 0.99, 'discount factor')
flags.DEFINE_float('learning_rate', 0.00025, 'The learning rate of training')

CURRENT_TIME=datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_SUMMARY_DIR = './summary/'
flags.DEFINE_string('summary_dir', BASE_SUMMARY_DIR+CURRENT_TIME+'/', 'summary dir')
BASE_MODEL_DIR = './model/'
flags.DEFINE_string('base_model_dir', BASE_MODEL_DIR, 'base model dir')
flags.DEFINE_string('model_dir', BASE_MODEL_DIR+CURRENT_TIME+'/', 'model dir')

# test super parameter, for small memory size and computation power computer
flags.DEFINE_integer('replay_memory_size', 10000, 'replay memory size')
flags.DEFINE_integer('max_steps', 100000, 'total train steps')
# real super parameter, for large memory size and computation power computer
#flags.DEFINE_integer('replay_memory_size', 1000000, 'replay_memory_size')
#flags.DEFINE_integer('max_steps', 50000000, 'total train steps')

flags.DEFINE_string('env_name', 'Breakout-v0', 'env_name')
# for restore parameter
MODEL_TIME='20170106-183605'
flags.DEFINE_string('restore_model_dir', BASE_MODEL_DIR+MODEL_TIME+'/', 'restored model dir')
flags.DEFINE_string('restore_summary_dir', BASE_SUMMARY_DIR+MODEL_TIME+'/', 'restored summary dir')
flags.DEFINE_string('gym_dir', './gym/'+MODEL_TIME+'/', 'gym dir')

flags.DEFINE_boolean('is_display', True, 'is_display')
flags.DEFINE_boolean('is_train', True, 'Train or play, need to restore_dir model when playing')
flags.DEFINE_boolean('is_restore_model', False, 'Restore model while training')

conf = flags.FLAGS

def main(_):
  agent = Agent(conf)
  if conf.is_train:
    agent.train()
  else:
    agent.play()

if __name__ == '__main__':
  tf.app.run()