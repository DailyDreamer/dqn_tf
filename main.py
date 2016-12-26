import tensorflow as tf
from datetime import datetime

from agent import Agent

flags = tf.app.flags
flags.DEFINE_integer('agent_history_length', 4, 'agent_history_length')

flags.DEFINE_string('env_name', 'Breakout-v0', 'env_name')
flags.DEFINE_integer('screen_height', 84, 'screen_height')
flags.DEFINE_integer('screen_width', 84, 'screen_width')
flags.DEFINE_integer('no_op_max', 30, 'no_op_max')

flags.DEFINE_integer('replay_memory_size', 1000000, 'replay_memory_size')

flags.DEFINE_integer('update_freq', 4, 'Q update freqency')
flags.DEFINE_integer('target_update_freq', 10000, 'target Q update freqency')
flags.DEFINE_integer('save_freq', 10000, 'model save freqency')

flags.DEFINE_float('init_ep', 1.0, 'init_ep')
flags.DEFINE_float('final_ep', 0.1, 'final_ep')
flags.DEFINE_integer('final_ep_frame', 1000000, 'final_ep_frame')

flags.DEFINE_integer('max_reward', 1, 'max_reward')
flags.DEFINE_integer('min_reward', -1, 'min_reward')

flags.DEFINE_float('discount', 0.99, 'discount factor')

TIME=datetime.now().strftime("%Y%m%d-%H%M%S")
flags.DEFINE_string('gym_dir', './gym/'+TIME, 'gym dir')
flags.DEFINE_string('summary_dir', './summary/'+TIME, 'summary dir')
flags.DEFINE_string('model_dir', './model/'+TIME+'/', 'model dir')

flags.DEFINE_integer('max_steps', 100000, 'total train steps')
#flags.DEFINE_integer('max_steps', 50000000, 'total train steps')
MODEL_TIME=''
BASESTEP=0
flags.DEFINE_string('restore_dir', './model/'+MODEL_TIME+'/'+str(BASESTEP)+'.ckpt', 'restore dir')
flags.DEFINE_integer('basestep', BASESTEP, 'base steps of restored model')

flags.DEFINE_boolean('is_display', True, 'is_display')
flags.DEFINE_boolean('is_train', True, 'Train or play, need to restore_dir model when playing')
flags.DEFINE_boolean('is_restore_model', FALSE, 'Restore model while training')


def main(_):
    agent = Agent(flags.FLAGS)
    if conf.is_train:
      agent.train(conf.t_train_max)
    else:
      agent.play(conf.final_ep)

if __name__ == '__main__':
  tf.app.run()