import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd
from IPython.display import clear_output
from tqdm import tqdm
from CompressionLibrary.utils import load_dataset, make_env, evaluate_adadeep, play_and_record_adadeep
from CompressionLibrary.CompressionTechniques import *
from CompressionLibrary.replay_buffer import ReplayBuffer
from CompressionLibrary.environments import *
from CompressionLibrary.custom_layers import ROIEmbedding, ROIEmbedding1D
from CompressionLibrary.reinforcement_models import DQNAgent
from CompressionLibrary import reward_functions
import sys
import gc
from datetime import datetime
from uuid import uuid4

eventid = datetime.now().strftime('%Y-%m-%d-%H-%M%S-') + str(uuid4())
datasets = ['cifar10']#['mnist', 'fashion_mnist']



current_state = 'layer_input'
next_state = 'layer_output'

base_model = 'vgg16'
eval_n_samples = 10
n_samples_mode = 256
training_batch_size = 512
rl_epochs = 1000
# layer_name_list = ['conv2d_1', 'dense_0', 'dense_1']
layer_name_list = [ 'block2_conv1',  'block3_conv2','fc1', 'fc2']
# layer_name_list = [ 'block2_conv1', 'block2_conv2', 
#                     'block3_conv1', 'block3_conv2', 'block3_conv3',
#                     'block4_conv1', 'block4_conv2', 'block4_conv3',
#                     'block5_conv1', 'block5_conv2', 'block5_conv3',
#                     'fc1', 'fc2']

generate_new_sample = play_and_record_adadeep
evaluate_agent = evaluate_adadeep
reward_func = reward_functions.reward_funcv1

logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler('/home/A00806415/DCC/ModelCompression/data/ModelCompression.log', 'w+'),
    logging.StreamHandler()],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')


filename_start = '+'.join(datasets)


print('/{}_{}_rw_history_{}_mode_{}.npy'.format(filename_start, eventid, eval_n_samples, n_samples_mode))


env = make_env(datasets[0], layer_name_list, current_state, next_state, base_model=base_model)
env.model.summary()

fc_state_dim = (1,)
fc_n_actions = len(env.dense_compressors)
conv_state_dim = list(env.get_state('current_state').shape)[1:]
print(conv_state_dim)
conv_n_actions = len(env.conv_compressors)

fc_agent = DQNAgent("dqn_agent_fc", fc_state_dim,
                    fc_n_actions, epsilon=0.9, layer_type='fc')
fc_target_network = DQNAgent(
    "target_network_fc", fc_state_dim, fc_n_actions, layer_type='fc')


conv_agent = DQNAgent("dqn_agent_conv", conv_state_dim,
                      conv_n_actions, epsilon=0.9, layer_type='cnn')
conv_target_network = DQNAgent(
    "target_network_conv", conv_state_dim, conv_n_actions, layer_type='cnn')


try:
    conv_agent.model.load_weights(
        './data/checkpoints/{}_my_checkpoint_conv'.format(filename_start))
    fc_agent.model.load_weights(
        './data/checkpoints/{}_my_checkpoint_fc'.format(filename_start))
except:
    print('Stored agent model could not be loaded.')
    pass


def load_weigths_into_target_network(agent, target_network):
    """ assign target_network.weights variables to their respective agent.weights values. """
    for i in range(len(agent.model.layers)):
        target_network.model.layers[i].set_weights(
            agent.model.layers[i].get_weights())


load_weigths_into_target_network(fc_agent, fc_target_network)
load_weigths_into_target_network(conv_agent, conv_target_network)

for w, w2 in zip(fc_agent.weights, fc_target_network.weights):
    tf.assert_equal(w, w2)
for w, w2 in zip(conv_agent.weights, conv_target_network.weights):
    tf.assert_equal(w, w2)

print("It works!")


def moving_average(x, span=100, **kw):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span, **kw).mean().values


mean_weights_history = []
mean_acc_history = []
mean_rw_history = []
td_loss_history_conv = []
td_loss_history_fc = []


def sample_batch(exp_replay, batch_size):
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(
        batch_size)
    return {
        'state': obs_batch,
        'action': act_batch,
        'rewards': reward_batch,
        'next_state': next_obs_batch,
        'done': is_done_batch,
    }


# @tf.function
def training_loop(state, action, rewards, next_state, done, agent, target_agent, loss_optimizer, n_actions, gamma=0.01):
    state = tf.cast(state, tf.float32)
    action = tf.cast(action, tf.int32)
    next_state = tf.cast(next_state, tf.float32)

    rewards = tf.cast(rewards, tf.float32)
    done = 1 - tf.cast(done, tf.float32)

    reference_qvalues = rewards + gamma * \
        tf.reduce_max(target_agent.get_qvalues(next_state), axis=1)
    reference_qvalues = reference_qvalues * (1 - done) - done

    masks = tf.one_hot(action, n_actions)
    with tf.GradientTape() as tape:
        q_values = agent.get_qvalues(state)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        td_loss = tf.reduce_mean((q_action - reference_qvalues) ** 2)

    gradients = tape.gradient(td_loss, agent.weights)
    loss_optimizer.apply_gradients(zip(gradients, agent.weights))
    return td_loss


fc_agent.epsilon = 0.999
conv_agent.epsilon = 0.999
epsilon_decay = 0.999
min_epsilon = 0.1
optimizer = tf.keras.optimizers.Adam(1e-5)

fc_exp_replay = ReplayBuffer(10 ** 5)
conv_exp_replay = ReplayBuffer(10 ** 5)

try:
    fc_exp_replay.load(
        './data/{}_adadeep_fc_replay_{}_{}.pkl'.format(filename_start, current_state, next_state))
    conv_exp_replay.load(
        './data/{}_adadeep_conv_replay_{}_{}.pkl'.format(filename_start, current_state, next_state))
    print('Succesfully loaded agent weights.')
except FileNotFoundError:
    pass

print('There are {} conv and {} fc instances.'.format(
    len(conv_exp_replay), len(fc_exp_replay)))

for dataset in datasets:
    env = make_env(dataset, layer_name_list, current_state, next_state, base_model=base_model)
    generate_new_sample(conv_agent, fc_agent, env, conv_exp_replay,
                            fc_exp_replay, dataset, current_state, next_state, n_samples_mode=n_samples_mode,reward_func=reward_func, n_steps=1, debug=True)

    print('There are {} conv and {} fc instances.'.format(
        len(conv_exp_replay), len(fc_exp_replay)))


dataset_counter = 0

with tqdm(total=rl_epochs,
          bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Epsilon: {postfix[0]:.4f} Last 3 RW: {postfix[1][0]:.2f}, {postfix[1][1]:.2f} & {postfix[1][2]:.2f} W: {postfix[2][0]:.2f}, {postfix[2][1]:.2f} & {postfix[2][2]:.2f} Acc: {postfix[3][0]:.2f}, {postfix[3][1]:.2f} & {postfix[3][2]:.2f}]",
          postfix=[conv_agent.epsilon, dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0})]) as t:
    for i in range(rl_epochs):
        # play
        generate_new_sample(conv_agent, fc_agent, env, conv_exp_replay,
                                fc_exp_replay, dataset, current_state, next_state, n_samples_mode=n_samples_mode,reward_func=reward_func, debug=False)

        # train fc
        batch_data = sample_batch(conv_exp_replay, batch_size=training_batch_size)
        batch_data['agent'] = conv_agent
        batch_data['target_agent'] = conv_target_network
        batch_data['loss_optimizer'] = optimizer
        batch_data['n_actions'] = conv_n_actions
        conv_loss_t = training_loop(**batch_data)
        td_loss_history_conv.append(conv_loss_t)

        # train
        batch_data = sample_batch(fc_exp_replay, batch_size=training_batch_size)
        batch_data['agent'] = fc_agent
        batch_data['target_agent'] = fc_target_network
        batch_data['loss_optimizer'] = optimizer
        batch_data['n_actions'] = fc_n_actions
        fc_loss_t = training_loop(**batch_data)
        td_loss_history_fc.append(fc_loss_t)

        # adjust agent parameters
        if i % 10 == 0:
            load_weigths_into_target_network(conv_agent, conv_target_network)
            conv_target_network.model.save_weights(
                './data/checkpoints/{}_my_checkpoint_conv'.format(filename_start))
            conv_agent.epsilon = max(conv_agent.epsilon * epsilon_decay, min_epsilon)
            t.postfix[0] = conv_agent.epsilon

            load_weigths_into_target_network(fc_agent, fc_target_network)
            fc_target_network.model.save_weights(
                './data/checkpoints/{}_my_checkpoint_fc'.format(filename_start))
            fc_agent.epsilon = conv_agent.epsilon

            
            rw, acc, weights = evaluate_agent(
                make_env(datasets[dataset_counter], layer_name_list, current_state, next_state, base_model=base_model), conv_agent, fc_agent, n_games=eval_n_samples, n_samples_mode=n_samples_mode ,reward_func=reward_func)
            mean_rw_history.append(rw)
            mean_acc_history.append(acc)
            mean_weights_history.append(weights)
            dataset_counter = (dataset_counter + 1) % len(datasets)

            with open('./data/numpy results/{}_{}_rw_history_{}_mode_{}.npy'.format(filename_start, eventid, eval_n_samples, n_samples_mode), 'wb') as f:
                np.save(f, np.array(mean_rw_history))
            with open('./data/numpy results/{}_{}_acc_history_{}_mode_{}.npy'.format(filename_start, eventid, eval_n_samples, n_samples_mode), 'wb') as f:
                np.save(f, np.array(mean_acc_history))
            with open('./data/numpy results/{}_{}_weights_history_{}_mode_{}.npy'.format(filename_start, eventid, eval_n_samples, n_samples_mode), 'wb') as f:
                np.save(f, np.array(mean_weights_history))
            with open('./data/numpy results/{}_{}_td_loss_fc_{}_mode_{}.npy'.format(filename_start, eventid, eval_n_samples, n_samples_mode), 'wb') as f:
                np.save(f, np.array(td_loss_history_fc))
            with open('./data/numpy results/{}_{}_td_loss_conv_{}_mode_{}.npy'.format(filename_start, eventid, eval_n_samples, n_samples_mode), 'wb') as f:
                np.save(f, np.array(td_loss_history_conv))

            t.postfix[1][2] = mean_rw_history[-1]
            try:
                t.postfix[1][1] = mean_rw_history[-2]
            except IndexError:
                t.postfix[1][1] = 0
            try:
                t.postfix[1][0] = mean_rw_history[-3]
            except IndexError:
                t.postfix[1][0] = 0

            t.postfix[2][2] = mean_weights_history[-1]
            try:
                t.postfix[2][1] = mean_weights_history[-2]
            except IndexError:
                t.postfix[2][1] = 0
            try:
                t.postfix[2][0] = mean_weights_history[-3]
            except IndexError:
                t.postfix[2][0] = 0

            t.postfix[3][2] = mean_acc_history[-1]
            try:
                t.postfix[3][1] = mean_acc_history[-2]
            except IndexError:
                t.postfix[3][1] = 0
            try:
                t.postfix[3][0] = mean_acc_history[-3]
            except IndexError:
                t.postfix[3][0] = 0
        t.update()
