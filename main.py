import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('agent_history_length', 4, 'agent_history_length')

flags.DEFINE_string('env_name', 'Breakout-v0', 'env_name')
flags.DEFINE_integer('screen_height', 84, 'screen_height')
flags.DEFINE_integer('screen_width', 84, 'screen_width')
flags.DEFINE_integer('no_op_max', 30, 'no_op_max')
flags.DEFINE_boolean('is_display', True, 'is_display')

flags.DEFINE_integer('replay_memory_size', 1000000, 'replay_memory_size')

flags.DEFINE_integer('update_freq', 4, 'update_freq')
flags.DEFINE_integer('target_update_freq', 10000, 'target_update_freq')
flags.DEFINE_integer('max_steps', 50000000, 'total train episodes')

flags.DEFINE_float('init_ep', 1.0, 'init_ep')
flags.DEFINE_float('final_ep', 0.1, 'final_ep')
flags.DEFINE_integer('final_ep_frame', 1000000, 'final_ep_frame')

flags.DEFINE_integer('max_reward', 1, 'max_reward')
flags.DEFINE_integer('min_reward', -1, 'min_reward')

flags.DEFINE_float('discount', 0.99, 'discount factor')

def main(_):