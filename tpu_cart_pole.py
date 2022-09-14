import tensorflow as tf
import logging

from CompressionLibrary.replay_buffer import ReplayBuffer


from uuid import uuid4
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import gym


run_id = datetime.now().strftime('%Y-%m-%d-%H-%M%S-') + str(uuid4())

try:
  # Use below for TPU
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
  tf.config.experimental_connect_to_cluster(resolver)
  # This is the TPU initialization code that has to be at the beginning.
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("All devices: ", tf.config.list_logical_devices('TPU'))
  strategy = tf.distribute.TPUStrategy(resolver)
  data_path = '/mnt/disks/mcdata/data'

except:
  print('ERROR: Not connected to a TPU runtime; Using GPU strategy instead!')
  strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
  data_path = './data'
  

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

epsilon_start_value = 0.99
epsilon_decay = 0.999
min_epsilon = 0.1
replay_buffer_size = 10 ** 5

rl_iterations = 100000
eval_n_samples = 10
batch_size_per_replica = 128
tuning_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
rl_batch_size = tuning_batch_size



class DQNAgent:
    def __init__(self, name, state_shape, n_actions, epsilon=0.0):
        """A simple DQN agent"""
        self.name = name

        input = tf.keras.layers.Input(shape=(state_shape))
        x = tf.keras.layers.Dense(512, activation='relu')(input)
        output = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.linear)(x)
        self.model = tf.keras.Model(inputs=input, outputs=output, name=name)
        self.epsilon = epsilon

    def get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        qvalues = self.model(state_t)
        return qvalues

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

def create_model():
    optimizer = tf.keras.optimizers.Adam(1e-5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()


    agent = DQNAgent('agent', state_shape=4, n_actions=2)
    agent.model.compile(optimizer=optimizer, loss=loss_object,
                    metrics=train_metric)

    return agent       

def make_env():
    return gym.make('CartPole-v0').env

def play_and_record(agent, env, exp_replay, n_steps=1):
    # initial state
    s = env.reset()
    rewards = 0
    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        qvalues = agent.get_qvalues(np.expand_dims(s, axis=0)).numpy()
        action = agent.sample_actions(qvalues=qvalues)[0]

        new_s, r, done, info = env.step(action)

        rewards += r
        exp_replay.add(s, action, r, new_s, done)
        s = new_s
        if done:
            s = env.reset()
        
    return rewards

def evaluate(env, agent, n_games=1, greedy=False, t_max=200):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues(np.expand_dims(s, axis=0)).numpy()
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, info = env.step(action)

            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)

env = make_env()


state_shape = env.observation_space.shape
n_actions = env.action_space.n

print(state_shape, n_actions)


def load_weigths_into_target_network(agent, target_network):
    """ assign target_network.weights variables to their respective agent.weights values. """
    for i in range(len(agent.model.layers)):
        target_network.model.layers[i].set_weights(
            agent.model.layers[i].get_weights())





def moving_average(x, span=100, **kw):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span, **kw).mean().values


mean_rw_history = []
td_loss_history = []


def sample_batch(exp_replay, batch_size):
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(
        batch_size)
    return (obs_batch, act_batch,reward_batch, next_obs_batch, is_done_batch)


@tf.function
def train_step(dataset_inputs):
    state, action, rewards, next_state, done = dataset_inputs
    state = tf.cast(state, tf.float32)
    action = tf.cast(action, tf.int32)
    next_state = tf.cast(next_state, tf.float32)

    rewards = tf.cast(rewards, tf.float32)
    done = 1 - tf.cast(done, tf.float32)

    # gamma 0.99
    reference_qvalues = rewards + 0.99 * \
        tf.reduce_max(target_agent.get_qvalues(next_state), axis=1)
    reference_qvalues = reference_qvalues * (1 - done) - done

    masks = tf.one_hot(action, n_actions)
    with tf.GradientTape() as tape:
        q_values = agent.get_qvalues(state)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        td_loss = tf.reduce_mean((q_action - reference_qvalues) ** 2)

    gradients = tape.gradient(td_loss, agent.model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, agent.model.trainable_weights))
    return td_loss

@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)


exp_replay = ReplayBuffer(replay_buffer_size)

target_agent = create_model()
with strategy.scope():
    
    agent = create_model()
    optimizer = tf.keras.optimizers.Adam(1e-5)


load_weigths_into_target_network(agent, target_agent)

for w, w2 in zip(agent.model.trainable_weights, target_agent.model.trainable_weights):
    tf.assert_equal(w, w2)
print("It works!")

    
agent.epsilon = epsilon_start_value

print('There are {} instances.'.format(len(exp_replay)))
play_and_record(agent, env, exp_replay, n_steps=1000)

print('There are {} instances.'.format(len(exp_replay)))

with tqdm(total=rl_iterations,
        bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Epsilon: {postfix[0]:.4f} Last 3 RW: {postfix[1][0]:.2f}, {postfix[1][1]:.2f} & {postfix[1][2]:.2f} ]",
        postfix=[agent.epsilon, dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0})]) as t:
    for i in range(rl_iterations):
        # generate new sample
        play_and_record(agent, env, exp_replay,n_steps=200)

            # train
        batch_data = sample_batch(exp_replay, batch_size=rl_batch_size)
        loss_t = distributed_train_step(batch_data)
        td_loss_history.append(loss_t)

        # adjust agent parameters
        if i % 500 == 0:
            load_weigths_into_target_network(agent, target_agent)
            agent.epsilon = max(agent.epsilon * epsilon_decay, min_epsilon)
            t.postfix[0] = agent.epsilon
            rw = evaluate(env, agent, n_games=10)            
            mean_rw_history.append(rw)

            t.postfix[1][2] = mean_rw_history[-1]
            try:
                t.postfix[1][1] = mean_rw_history[-2]
            except IndexError:
                t.postfix[1][1] = 0
            try:
                t.postfix[1][0] = mean_rw_history[-3]
            except IndexError:
                t.postfix[1][0] = 0
        t.update()