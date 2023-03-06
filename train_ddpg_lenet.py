import tensorflow as tf
import logging

from CompressionLibrary.reinforcement_models import DDPGWeights2D
from CompressionLibrary.replay_buffer import ReplayBuffer
from CompressionLibrary.environments import ModelCompressionSVDEnv
import numpy as np

from uuid import uuid4
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_datasets as tfds

import gc


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


run_id = datetime.now().strftime('%Y-%m-%d-%H-%M%S-') + str(uuid4())

dataset_name = 'mnist'
data_path = './data'
  
logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(f'/home/A00806415/DCC/ModelCompression/data/ModelCompression_DDPG_{dataset_name}.log', 'w+')],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)



exploration_filename = data_path+'/training_exploration_DDPG.csv'
test_filename = data_path+'/test_evaluate_DDPG.csv'
agents_path = data_path+'/checkpoints/{}_my_checkpoint_DDPG_'.format(dataset_name)

current_state = 'layer_weights'
next_state = 'layer_weights'

epsilon_start_value = 0.9
epsilon_decay = 0.995
min_epsilon = 0.1
replay_buffer_size = 5000

rl_iterations = 2000
n_samples_mode = -1
strategy = None
tuning_batch_size = 64
tuning_epochs = 0
rl_batch_size = 64
verbose = 0
tuning_mode = 'layer'

layer_name_list = ['conv2d_1',  'dense', 'dense_1']

def dataset_preprocessing(img, label):
    img = tf.cast(img, tf.float32)
    img = img/255.0
    return img, label

def load_dataset(batch_size=128):
    splits, info = tfds.load(dataset_name, as_supervised=True, with_info=True, shuffle_files=True, 
                                split=['train[:80%]', 'train[80%:]','test'])

    (train_examples, validation_examples, test_examples) = splits
    num_examples = info.splits['train'].num_examples

    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape

    input_shape = (28,28,1)

    train_ds = train_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = validation_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, test_ds, input_shape, num_classes

train_ds, valid_ds, test_ds, input_shape, _ = load_dataset(tuning_batch_size)

def create_model():
    checkpoint_path = f"./data/models/lenet_{dataset_name}/cp.ckpt"
    optimizer = tf.keras.optimizers.Adam(1e-5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    input = tf.keras.layers.Input((28,28,1))
    x = tf.keras.layers.Conv2D(6, (5,5), padding='SAME', activation='sigmoid', name='conv2d')(input)
    x = tf.keras.layers.AveragePooling2D((2,2), strides=2, name='avg_pool_1')(x)
    x = tf.keras.layers.Conv2D(16, (5,5), padding='VALID', activation='sigmoid', name='conv2d_1')(x)
    x = tf.keras.layers.AveragePooling2D((2,2), strides=2, name='avg_pool_2')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(120, activation='sigmoid', name='dense')(x)
    x = tf.keras.layers.Dense(84, activation='sigmoid', name='dense_1')(x)
    x = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(input, x, name='LeNet')
    model.compile(optimizer=optimizer, loss=loss_object,
                    metrics=[train_metric])

    try:
        model.load_weights(checkpoint_path).expect_partial()
    except:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)
        model.fit(train_ds,
          epochs=3000,
          validation_data=valid_ds,
          callbacks=[cp_callback])

    return model            


input_shape = (28,28,1)

def make_env(create_model, train_ds, valid_ds, test_ds, input_shape, layer_name_list, num_feature_maps, tuning_batch_size, tuning_epochs, verbose=0, tuning_mode='layer', current_state_source='layer_input', next_state_source='layer_output', strategy=None, model_path='./data'):

    w_comprs = ['InsertDenseSVD'] 
    l_comprs = ['MLPCompression']
    compressors_list = w_comprs +  l_comprs

    parameters = {}
    parameters['InsertDenseSVD'] = {'layer_name': None, 'percentage': None}
    parameters['MLPCompression'] = {'layer_name': None, 'percentage': None}

    env = ModelCompressionSVDEnv(compressors_list, create_model, parameters, train_ds, valid_ds, test_ds, layer_name_list, input_shape, tuning_batch_size=tuning_batch_size, tuning_epochs=tuning_epochs, current_state_source=current_state_source, next_state_source=next_state_source, num_feature_maps=num_feature_maps, verbose=verbose,strategy=strategy, model_path=model_path)

    return env

env = make_env(
        create_model=create_model, 
        train_ds=train_ds, 
        valid_ds=valid_ds, 
        test_ds=test_ds, 
        input_shape=input_shape, 
        layer_name_list=layer_name_list, 
        num_feature_maps=n_samples_mode,
        tuning_batch_size=tuning_batch_size,
        tuning_epochs = tuning_epochs, 
        verbose=verbose, 
        tuning_mode='final', 
        current_state_source=current_state, 
        next_state_source=next_state, 
        strategy=strategy, 
        model_path=data_path)

env.model.summary()

conv_shape, dense_shape = env.observation_space()
action_space = env.action_space()
num_actions = 1

fc_n_actions = conv_n_actions = num_actions
print(conv_shape, dense_shape)

upper_bound = action_space['high']
lower_bound = action_space['low']

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


conv_agent = DDPG("ddpg_agent_conv", conv_shape,
                        fc_n_actions, epsilon=epsilon_start_value, layer_type='conv')
conv_target_agent = DDPG(
    "target_network_fc", conv_shape,
                        fc_n_actions, epsilon=epsilon_start_value, layer_type='conv')

conv_agent.actor.summary()
conv_agent.critic.summary()



try:
    conv_target_agent.actor.load_weights(agents_path+'actor')
    conv_target_agent.critic.load_weights(agents_path+'critic')
except:
    print('Failed to find pretrained models for the RL agents.')
    pass

def sample_batch(exp_replay, batch_size):
    obs_batch, act_batch, reward_batch, next_obs_batch, _ = exp_replay.sample(
        batch_size)
    return {
        'state_batch': tf.cast(obs_batch.to_tensor(), tf.float32),
        'action_batch': tf.cast(act_batch, tf.int32),
        'reward_batch': tf.cast(reward_batch, tf.float32),
        'next_state_batch': tf.cast(next_obs_batch.to_tensor(), tf.float32)
    }

@tf.function(experimental_relax_shapes=True)
def update_fc(
    state_batch,
    action_batch,
    reward_batch,
    next_state_batch,
):
    # Training and updating Actor & Critic networks.
    # See Pseudo Code.
    with tf.GradientTape() as tape:
        target_actions = conv_target_agent.actor(next_state_batch, training=True)
        y = reward_batch + gamma * conv_target_agent.critic(
            [next_state_batch, target_actions], training=True
        )
        critic_value = conv_agent.critic([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, conv_agent.critic.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, conv_agent.critic.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = conv_agent.actor(state_batch, training=True)
        critic_value = conv_agent.critic([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, conv_agent.actor.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, conv_agent.actor.trainable_variables)
    )

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def copy_weights(target, agent):
    target.actor.set_weights(agent.actor.get_weights())
    target.critic.set_weights(agent.critic.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
buffer_conv = ReplayBuffer(replay_buffer_size)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

def play_and_record(conv_agent, env, fc_replay, run_id, test_number, dataset_name, save_name, n_games=10, exploration=True):
    # initial state
    s = env.reset()
    # Play the game for n_steps as per instructions above

    logger = logging.getLogger(__name__)
    rewards = []
    acc = []
    weights = []
    infos = []
    total_time = 0

    for it in range(n_games):
        start = datetime.now()
        for k in range(1, len(env.layer_name_list)+1):
            tf.keras.backend.clear_session()
            # Get the current layer name
            current_layer_name = env.layer_name_list[env._layer_counter]

            # Action is the mode of the action.
            action = conv_agent.policy(s, exploration=exploration)
            logger.debug(f'Action for layer {current_layer_name} layer is {action}')

            # Apply action
            new_s, r, done, info = env.step(action)


            logger.debug(f'Iteration {it} - Layer {current_layer_name} {k}/{len(env.layer_name_list)}\tChosen action {action} has {r} reward.')
            logger.debug(info)

            row = {'state': s, 'action': action, 'reward': r,
                   'next_state': new_s, 'done': done, 'info': info}


            num_inst = s.shape[0]

            if exploration:
                logging.debug('Storing instance in fc replay.')
                fc_replay.add_multiple(s, [action]*num_inst, [info['reward_step']]*num_inst, new_s, [None]*num_inst)
                logging.debug('Sampling instances from fc replay.')
                batch = sample_batch(fc_replay, rl_batch_size)
                logging.debug('Training fc agent.')
                update_fc(**batch)
                logging.debug('Updating weights of target agent.')
                update_target(conv_target_agent.actor.variables, conv_agent.actor.variables, tau)
                update_target(conv_target_agent.critic.variables, conv_agent.critic.variables, tau)


            s = env.get_state('current_state')

            if done:
                s = env.reset()
                conv_target_agent.actor.save_weights(agents_path+'fc_actor')
                conv_target_agent.critic.save_weights(agents_path+'fc_critic')
                break

            gc.collect()
        

        actions = info['actions']
        info['actions'] = ','.join(map(lambda x: str(int(x*100)), actions))
        info['run_id'] = run_id
        info['test_number'] = test_number
        info['game_id'] = it
        info['dataset'] = dataset_name
        del info['layer_name']
        reward = info['reward_all_steps']
        rewards.append(reward)
        acc.append(info['test_acc_after'])
        weights.append(info['weights_after'])
        new_row = pd.DataFrame(info, index=[0])
        new_row.to_csv(save_name, mode='a', index=False)

        # Correct reward is the last value of r.
        rewards.append(r)
        end = datetime.now()
        time_diff = (end - start).total_seconds()
        total_time += time_diff
        logger.info(f'Took {time_diff} seconds for one compression.')

    logger.info(f'Evaluation of {n_games} took {total_time} secs. An average of {total_time/n_games} secs per game.')

    return np.mean(rewards), np.mean(acc), np.mean(weights)

mean_weights_history = np.empty(shape=rl_iterations+1)
mean_acc_history = np.empty(shape=rl_iterations+1)
mean_rw_history = np.empty(shape=rl_iterations+1)

mean_weights_history[0] = env.weights_before
mean_acc_history[0] = env.test_acc_before
mean_rw_history[0] = 0.0

with tqdm(total=rl_iterations,
        bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Last 3 RW: {postfix[0][0]:.2f}, {postfix[0][1]:.2f} & {postfix[0][2]:.2f} W: {postfix[1][0]:.2f}, {postfix[1][1]:.2f} & {postfix[1][2]:.2f} Acc: {postfix[2][0]:.2f}, {postfix[2][1]:.2f} & {postfix[2][2]:.2f}] Replay: {postfix[3]} .",
        postfix=[dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0}), len(buffer_conv)]) as t:
    for i in range(1, rl_iterations+1):
        rw, acc, weights = play_and_record(conv_agent, env, buffer_conv, run_id=run_id, test_number=i, dataset_name=dataset_name,save_name=exploration_filename,n_games=1, exploration=True)

        rw, acc, weights = play_and_record(conv_agent, env, buffer_conv, run_id=run_id, test_number=i, dataset_name=dataset_name,save_name=test_filename,n_games=1, exploration=False)
        mean_rw_history[i] = rw
        mean_acc_history[i] = acc
        mean_weights_history[i] = weights

        t.postfix[0][2] = mean_rw_history[i]
        t.postfix[3] = len(buffer_conv)
        
        try:
            t.postfix[0][1] = mean_rw_history[i-1]
        except IndexError:
            t.postfix[0][1] = 0
        try:
            t.postfix[0][0] = mean_rw_history[i-2]
        except IndexError:
            t.postfix[0][0] = 0

        t.postfix[1][2] = mean_weights_history[i]
        try:
            t.postfix[1][1] = mean_weights_history[i-1]
        except IndexError:
            t.postfix[1][1] = 0
        try:
            t.postfix[1][0] = mean_weights_history[i-2]
        except IndexError:
            t.postfix[1][0] = 0

        t.postfix[2][2] = mean_acc_history[i]
        try:
            t.postfix[2][1] = mean_acc_history[i-1]
        except IndexError:
            t.postfix[2][1] = 0
        try:
            t.postfix[2][0] = mean_acc_history[i-2]
        except IndexError:
            t.postfix[2][0] = 0



        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.title.set_text('Accuracy')
        ax1.plot(mean_acc_history)
        ax2.title.set_text('Weights')
        ax2.plot(mean_weights_history)
        ax3.title.set_text('Reward')
        ax3.plot(mean_rw_history)
        plt.xlabel('Epochs')
        plt.savefig(f'./data/figures/DDPG_test_stats.png')
        plt.close()

        t.update()