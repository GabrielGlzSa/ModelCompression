import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd
from IPython.display import clear_output
from tqdm import tqdm
import tensorflow_datasets as tfds

from CompressionLibrary.agent_evaluators import make_env_imagenet, evaluate_agents, play_and_record
from CompressionLibrary.CompressionTechniques import *
from CompressionLibrary.replay_buffer import ReplayBuffer
from CompressionLibrary.environments import *
from CompressionLibrary.custom_layers import ROIEmbedding, ROIEmbedding1D
from CompressionLibrary.reinforcement_models import DQNAgent
from CompressionLibrary import reward_functions
from keras.applications import imagenet_utils
import sys
import gc
from datetime import datetime
from uuid import uuid4

import tracemalloc
tracemalloc.start()


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

eventid = datetime.now().strftime('%Y-%m-%d-%H-%M%S-') + str(uuid4())
dataset = 'imagenet2012'
current_state = 'layer_input'
next_state = 'layer_output'

eval_n_samples = 10
n_samples_mode = 128
rl_batch_size = 128
tuning_batch_size = 64

iterations = 1000
epsilon_decay = 0.999
min_epsilon = 0.1
fc_exp_replay = ReplayBuffer(10 ** 4)
conv_exp_replay = ReplayBuffer(10 ** 4)


layer_name_list = [ 'block2_conv1', 'block2_conv2', 
                    'block3_conv1', 'block3_conv2', 'block3_conv3',
                    'block4_conv1', 'block4_conv2', 'block4_conv3',
                    'block5_conv1', 'block5_conv2', 'block5_conv3',
                    'fc1', 'fc2']



def resize_image(image, shape = (224,224)):
  target_width = shape[0]
  target_height = shape[1]
  initial_width = tf.shape(image)[0]
  initial_height = tf.shape(image)[1]
  im = image
  ratio = 0
  if(initial_width < initial_height):
    ratio = tf.cast(256 / initial_width, tf.float32)
    h = tf.cast(initial_height, tf.float32) * ratio
    im = tf.image.resize(im, (256, h), method="bicubic")
  else:
    ratio = tf.cast(256 / initial_height, tf.float32)
    w = tf.cast(initial_width, tf.float32) * ratio
    im = tf.image.resize(im, (w, 256), method="bicubic")
  width = tf.shape(im)[0]
  height = tf.shape(im)[1]
  startx = width//2 - (target_width//2)
  starty = height//2 - (target_height//2)
  im = tf.image.crop_to_bounding_box(im, startx, starty, target_width, target_height)
  return im

@tf.function
def imagenet_preprocessing(img, label):
    img = tf.cast(img, tf.float32)
    img = resize_image(img)
    img = tf.keras.applications.vgg16.preprocess_input(img, data_format=None)
    return img, label

splits, info = tfds.load('imagenet2012', as_supervised=True, with_info=True, shuffle_files=True, 
                            split=['train[:80%]', 'train[80%:]', 'test'], data_dir='./data')

(train_examples, validation_examples, test_examples) = splits
num_examples = info.splits['train'].num_examples

num_classes = info.features['label'].num_classes
input_shape = info.features['image'].shape


optimizer = tf.keras.optimizers.Adam(1e-5)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_metric = tf.keras.metrics.SparseCategoricalAccuracy()


model = tf.keras.applications.vgg16.VGG16(
                        include_top=True,
                        weights='imagenet',
                        input_shape=(224,224,3),
                        classes=num_classes,
                        classifier_activation='softmax'
                    )
model.compile(optimizer=optimizer, loss=loss_object,
                metrics=train_metric)

model.summary()            
train_ds = train_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(tuning_batch_size).prefetch(tf.data.AUTOTUNE)
valid_ds = validation_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(tuning_batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(tuning_batch_size).prefetch(tf.data.AUTOTUNE)


# logging.basicConfig(level=logging.DEBUG, handlers=[
#     logging.FileHandler('/home/A00806415/DCC/ModelCompression/data/ModelCompression.log', 'w+'),#],
#     logging.StreamHandler()],
#     format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')

input_shape = (224,224,3)
env = make_env_imagenet(dataset, train_ds, valid_ds, test_ds, input_shape, num_classes, layer_name_list, n_samples_mode, tuning_batch_size, current_state, next_state)
env.model.summary()

conv_shape, dense_shape = env.observation_space()
conv_n_actions, fc_n_actions = env.action_space()

print(conv_shape, dense_shape)


fc_agent = DQNAgent("dqn_agent_fc", dense_shape,
                    fc_n_actions, epsilon=0.9, layer_type='fc')
fc_target_network = DQNAgent(
    "target_network_fc", dense_shape, fc_n_actions, layer_type='fc')


conv_agent = DQNAgent("dqn_agent_conv", conv_shape,
                      conv_n_actions, epsilon=0.9, layer_type='cnn')
conv_target_network = DQNAgent(
    "target_network_conv", conv_shape, conv_n_actions, layer_type='cnn')


try:
    conv_target_network.model.load_weights(
        './data/checkpoints/{}_my_checkpoint_conv'.format(dataset))
    fc_target_network.model.load_weights(
        './data/checkpoints/{}_my_checkpoint_fc'.format(dataset))
except:
    print('Failed to find pretrained models for the RL agents.')
    pass


def load_weigths_into_target_network(agent, target_network):
    """ assign target_network.weights variables to their respective agent.weights values. """
    for i in range(len(agent.model.layers)):
        target_network.model.layers[i].set_weights(
            agent.model.layers[i].get_weights())


load_weigths_into_target_network(fc_agent, fc_target_network)
load_weigths_into_target_network(conv_agent, conv_target_network)

for w, w2 in zip(fc_agent.model.trainable_weights, fc_target_network.model.trainable_weights):
    tf.assert_equal(w, w2)
for w, w2 in zip(conv_agent.model.trainable_weights, conv_target_network.model.trainable_weights):
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


@tf.function
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

optimizer = tf.keras.optimizers.Adam(1e-5)

snapshot1 = tracemalloc.take_snapshot()

print('There are {} conv and {} fc instances.'.format(len(conv_exp_replay), len(fc_exp_replay)))
play_and_record(conv_agent, fc_agent, env, conv_exp_replay, fc_exp_replay, n_steps=1)

print('There are {} conv and {} fc instances.'.format(len(conv_exp_replay), len(fc_exp_replay)))


# evaluate_agents(env, conv_agent, fc_agent, n_games=eval_n_samples, save_name='./data/test_evaluate.csv')

snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)

top_stats = snapshot2.compare_to(snapshot1, 'lineno')

print("[ Top 10 differences ]")
for stat in top_stats[:10]:
    print(stat)

# with tqdm(total=iterations,
#           bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Epsilon: {postfix[0]:.4f} Last 3 RW: {postfix[1][0]:.2f}, {postfix[1][1]:.2f} & {postfix[1][2]:.2f} W: {postfix[2][0]:.2f}, {postfix[2][1]:.2f} & {postfix[2][2]:.2f} Acc: {postfix[3][0]:.2f}, {postfix[3][1]:.2f} & {postfix[3][2]:.2f}]",
#           postfix=[conv_agent.epsilon, dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0})]) as t:
#     for i in range(iterations):
#         # play
#         generate_new_sample(conv_agent, fc_agent, env, conv_exp_replay,
#                                 fc_exp_replay, dataset, current_state, next_state, n_samples_mode=n_samples_mode,reward_func=reward_func, debug=False)

#         # train fc
#         batch_data = sample_batch(conv_exp_replay, batch_size=training_batch_size)
#         batch_data['agent'] = conv_agent
#         batch_data['target_agent'] = conv_target_network
#         batch_data['loss_optimizer'] = optimizer
#         batch_data['n_actions'] = conv_n_actions
#         conv_loss_t = training_loop(**batch_data)
#         td_loss_history_conv.append(conv_loss_t)

#         # train
#         batch_data = sample_batch(fc_exp_replay, batch_size=training_batch_size)
#         batch_data['agent'] = fc_agent
#         batch_data['target_agent'] = fc_target_network
#         batch_data['loss_optimizer'] = optimizer
#         batch_data['n_actions'] = fc_n_actions
#         fc_loss_t = training_loop(**batch_data)
#         td_loss_history_fc.append(fc_loss_t)

#         # adjust agent parameters
#         if i % 10 == 0:
#             load_weigths_into_target_network(conv_agent, conv_target_network)
#             conv_target_network.model.save_weights(
#                 './data/checkpoints/{}_my_checkpoint_conv'.format(dataset))
#             conv_agent.epsilon = max(conv_agent.epsilon * epsilon_decay, min_epsilon)
#             t.postfix[0] = conv_agent.epsilon

#             load_weigths_into_target_network(fc_agent, fc_target_network)
#             fc_target_network.model.save_weights(
#                 './data/checkpoints/{}_my_checkpoint_fc'.format(dataset))
#             fc_agent.epsilon = max(fc_agent.epsilon * epsilon_decay, min_epsilon)

#             rw, acc, weights = evaluate_agent(
#                 make_env_preloaded_dataset(dataset, train_ds, valid_ds, test_ds, input_shape, num_classes, layer_name_list, current_state, next_state), conv_agent, fc_agent, n_games=eval_n_samples, n_samples_mode=n_samples_mode ,reward_func=reward_func)
#             mean_rw_history.append(rw)
#             mean_acc_history.append(acc)
#             mean_weights_history.append(weights)

#             # with open('./data/numpy results/{}_{}_rw_history_{}_mode_{}.npy'.format(dataset, eventid, eval_n_samples, n_samples_mode), 'wb') as f:
#             #     np.save(f, np.array(mean_rw_history))
#             # with open('./data/numpy results/{}_{}_acc_history_{}_mode_{}.npy'.format(dataset, eventid, eval_n_samples, n_samples_mode), 'wb') as f:
#             #     np.save(f, np.array(mean_acc_history))
#             # with open('./data/numpy results/{}_{}_weights_history_{}_mode_{}.npy'.format(dataset, eventid, eval_n_samples, n_samples_mode), 'wb') as f:
#             #     np.save(f, np.array(mean_weights_history))
#             # with open('./data/numpy results/{}_{}_td_loss_fc_{}_mode_{}.npy'.format(dataset, eventid, eval_n_samples, n_samples_mode), 'wb') as f:
#             #     np.save(f, np.array(td_loss_history_fc))
#             # with open('./data/numpy results/{}_{}_td_loss_conv_{}_mode_{}.npy'.format(dataset, eventid, eval_n_samples, n_samples_mode), 'wb') as f:
#             #     np.save(f, np.array(td_loss_history_conv))

#             t.postfix[1][2] = mean_rw_history[-1]
#             try:
#                 t.postfix[1][1] = mean_rw_history[-2]
#             except IndexError:
#                 t.postfix[1][1] = 0
#             try:
#                 t.postfix[1][0] = mean_rw_history[-3]
#             except IndexError:
#                 t.postfix[1][0] = 0

#             t.postfix[2][2] = mean_weights_history[-1]
#             try:
#                 t.postfix[2][1] = mean_weights_history[-2]
#             except IndexError:
#                 t.postfix[2][1] = 0
#             try:
#                 t.postfix[2][0] = mean_weights_history[-3]
#             except IndexError:
#                 t.postfix[2][0] = 0

#             t.postfix[3][2] = mean_acc_history[-1]
#             try:
#                 t.postfix[3][1] = mean_acc_history[-2]
#             except IndexError:
#                 t.postfix[3][1] = 0
#             try:
#                 t.postfix[3][0] = mean_acc_history[-3]
#             except IndexError:
#                 t.postfix[3][0] = 0
#         t.update()

#         if i % 10 == 0:
#             clear_output(True)
#             fig = plt.figure(figsize=(16, 12))
#             plt.subplot(1, 4, 1)
#             plt.title("Mean reward per game")
#             plt.plot(mean_rw_history)
#             plt.grid()

#             assert not np.isnan(td_loss_history_fc).any(
#             ) and not np.isnan(td_loss_history_conv).any()

#             plt.subplot(1, 4, 2)
#             plt.title("TD loss history (moving average)")
#             plt.plot(moving_average(
#                 np.array(td_loss_history_conv), span=10, min_periods=10))
#             plt.plot(moving_average(
#                 np.array(td_loss_history_fc), span=10, min_periods=10))
#             plt.grid()

#             plt.subplot(1, 4, 3)
#             plt.title("Weights history")
#             plt.plot(mean_weights_history)
#             plt.grid()

#             plt.subplot(1, 4, 4)
#             plt.title("Acc history")
#             plt.plot(mean_acc_history)
#             plt.grid()
#             plt.savefig('./data/stats/{}_{}_stats_plot_{}_mode_{}.png'.format('_'.join(dataset), eventid, eval_n_samples, n_samples_mode))
#             plt.close(fig)