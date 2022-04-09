import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from utils import load_dataset
from CompressionTechniques import *
from replay_buffer import ReplayBuffer
from environments import LayerEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')

class DQNAgent:
    def __init__(self, name, state_shape, n_actions, epsilon=0.0):
        """A simple DQN agent"""

        features = tf.keras.layers.Input(shape=state_shape)
        x = tf.keras.layers.Dense(512, activation='relu')(features)
        output = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.linear)(x)
        self.model = tf.keras.Model(inputs=features, outputs=output)
        self.model.summary()
        self.weights = self.model.trainable_weights
        self.epsilon = epsilon

    def get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues = self.model(state_t)
        return qvalues

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

def evaluate(env, agent, n_games=1, greedy=True, t_max=10000):
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

def play_and_record(agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time
    """
    # initial state
    s = env.reset()
    rewards = 0

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        qvalues = agent.get_qvalues(np.expand_dims(s, axis=0)).numpy()
        action = agent.sample_actions(qvalues=qvalues)[0]

        new_s, r, done, info = env.step(action)
        rewards += r
        if info['action_overwritten']:
            action = 0

        print(s, action, r, new_s, done, info)
        exp_replay.add(s, action, r, new_s, done)
        s = new_s
        if done:
            s = env.reset()

    return rewards

def make_env():
    train_ds, val_ds, test_ds, input_shape, num_classes = load_dataset('mnist')
    train_ds, valid_ds, test_ds, input_shape, num_classes = load_dataset('mnist')
    # optimizer = tf.keras.optimizers.Adam()
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    # model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_0',
    #                                                     input_shape=input_shape),
    #                              tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
    #                              tf.keras.layers.MaxPool2D((2, 2), 2),
    #                              tf.keras.layers.Flatten(),
    #                              tf.keras.layers.Dense(128, activation='relu', name='dense_0'),
    #                              tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
    #                              tf.keras.layers.Dense(num_classes, activation='softmax', name='dense_softmax')
    #                              ])
    # model.compile(optimizer=optimizer, loss=loss_object, metrics=train_metric)
    # model.fit(train_ds, epochs=5, validation_data=valid_ds)
    model_path = './data/full_model/test'
    # model.save(model_path)

    # compressors_list = ['DepthwiseSeparableConvolution', 'FireLayerCompression', 'InsertSVDConv',
    #                    'SparseConnectionsCompression']
    compressors_list = ['DeepCompression', 'InsertDenseSVD', 'InsertDenseSVDCustom', 'InsertDenseSparse',
                       'ReplaceDenseWithGlobalAvgPool']
    # compressors_list = ['DepthwiseSeparableConvolution', 'FireLayerCompression', 'InsertSVDConv',
    #                    'SparseConnectionsCompression','DeepCompression', 'InsertDenseSVD', 'InsertDenseSVDCustom', 'InsertDenseSparse',
    #                    'ReplaceDenseWithGlobalAvgPool']

    parameters = {}
    parameters['DeepCompression'] = {'layer_name': 'dense_0', 'threshold': 0.001}
    parameters['ReplaceDenseWithGlobalAvgPool'] = {'layer_name': 'dense_1'}
    parameters['InsertDenseSVD'] = {'layer_name': 'dense_0', 'units': 16}
    parameters['InsertDenseSVDCustom'] = {'layer_name': 'dense_0', 'units': 16}
    parameters['InsertDenseSparse'] = {'layer_name': 'dense_0', 'verbose': True, 'units': 16}
    parameters['InsertSVDConv'] = {'layer_name': 'conv2d_1', 'units': 8}
    parameters['DepthwiseSeparableConvolution'] = {'layer_name': 'conv2d_1'}
    parameters['FireLayerCompression'] = {'layer_name': 'conv2d_1'}
    # parameters['MLPCompression'] = {'layer_name': 'conv2d_1'}
    parameters['SparseConnectionsCompression'] = {'layer_name': 'conv2d_1', 'epochs': 20,
                                                  'target_perc': 0.75, 'conn_perc_per_epoch': 0.1}

    layer_name_list = ['dense_0', 'dense_1']
    env = LayerEnv(compressors_list, model_path, parameters,
                 train_ds, val_ds,
                 layer_name_list)

    return env

env = make_env()
env.model.summary()
state_dim = env.observation_space()
n_actions = env.action_space()

agent = DQNAgent("dqn_agent", state_dim, n_actions, epsilon=0.9)
target_network = DQNAgent("target_network", state_dim, n_actions)

def load_weigths_into_target_network(agent, target_network):
    """ assign target_network.weights variables to their respective agent.weights values. """
    for i in range(len(agent.model.layers)):
        target_network.model.layers[i].set_weights(agent.model.layers[i].get_weights())


load_weigths_into_target_network(agent, target_network)

for w, w2 in zip(agent.weights, target_network.weights):
    tf.assert_equal(w, w2)
print("It works!")


def moving_average(x, span=100, **kw):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span, **kw).mean().values


mean_rw_history = []
td_loss_history = []




def sample_batch(exp_replay, batch_size):
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)
    return {
        'state': obs_batch,
        'action': act_batch,
        'rewards': reward_batch,
        'next_state': next_obs_batch,
        'done': is_done_batch,
    }


@tf.function
def training_loop(state, action, rewards, next_state, done, agent, target_agent, loss_optimizer, gamma=0.99):
    state = tf.cast(state, tf.float32)
    action = tf.cast(action, tf.int32)
    next_state = tf.cast(next_state, tf.float32)

    rewards = tf.cast(rewards, tf.float32)
    done = 1 - tf.cast(done, tf.float32)

    reference_qvalues = rewards + gamma * tf.reduce_max(target_agent.get_qvalues(next_state), axis=1)
    reference_qvalues = reference_qvalues * (1 - done) - done

    masks = tf.one_hot(action, n_actions)
    with tf.GradientTape() as tape:
        q_values = agent.get_qvalues(state)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        td_loss = tf.reduce_mean((q_action - reference_qvalues) ** 2)

    gradients = tape.gradient(td_loss, agent.weights)
    loss_optimizer.apply_gradients(zip(gradients, agent.weights))
    return td_loss


agent.epsilon = 0.9
min_epsilon = 0.1
optimizer = tf.keras.optimizers.Adam(1e-5)
iterations = 100

exp_replay = ReplayBuffer(10 ** 4)
play_and_record(agent, env, exp_replay, n_steps=100)

with tqdm.tqdm(total=iterations,
          bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix[0]}: {postfix[1]:.4f} {postfix[2]}: {postfix[3][0]:.2f}, {postfix[3][1]:.2f} & {postfix[3][2]:.2f}]",
          postfix=["Epsilon", agent.epsilon, 'Last 3 RW', dict({0:0,1:0,2:0})]) as t:
    for i in range(iterations):
        # play
        play_and_record(agent, env, exp_replay, 30)

        # train
        batch_data = sample_batch(exp_replay, batch_size=64)
        batch_data['agent'] = agent
        batch_data['target_agent'] = target_network
        batch_data['loss_optimizer'] = optimizer
        loss_t = training_loop(**batch_data)
        td_loss_history.append(loss_t)

        # adjust agent parameters
        if i % 500 == 0:
            load_weigths_into_target_network(agent, target_network)
            target_network.model.save_weights('./checkpoints/my_checkpoint')
            agent.epsilon = max(agent.epsilon * 0.98, min_epsilon)
            t.postfix[1] = agent.epsilon
            mean_rw_history.append(evaluate(make_env(), agent, n_games=3))
            t.postfix[3][2] = mean_rw_history[-1]
            try:
                t.postfix[3][1] = mean_rw_history[-2]
            except IndexError:
                t.postfix[3][1] = 0
            try:
                t.postfix[3][0] = mean_rw_history[-3]
            except IndexError:
                t.postfix[3][0] = 0

        if np.mean(mean_rw_history[-10:]) > 1.7:
            print("That's good enough for tutorial.")
            break
        t.update()



plt.subplot(1, 2, 1)
plt.title("mean reward per game")
plt.plot(mean_rw_history)
plt.grid()

assert not np.isnan(loss_t)

plt.subplot(1, 2, 2)
plt.title("TD loss history (moving average)")
plt.plot(moving_average(np.array(td_loss_history), span=100, min_periods=100))
plt.grid()
plt.show()