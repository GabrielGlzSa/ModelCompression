import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import logging

from CompressionLibrary.reinforcement_models import DDPGWeights2D
from CompressionLibrary.replay_buffer import ReplayBufferMultipleDatasets
from CompressionLibrary.environments import ModelCompressionSVDEnv
import numpy as np

from uuid import uuid4
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_datasets as tfds
from functools import partial
import gc


dataset_names = ['fashion_mnist', 'kmnist']
log_name = '-'.join(dataset_names)
logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(f'/home/A00806415/DCC/ModelCompression/data/ModelCompression_DDPG_{log_name}.log', 'w+')],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
logger.info('Found GPU at: {}'.format(device_name))

run_id = datetime.now().strftime('%Y-%m-%d-%H-%M%S-') + str(uuid4())

data_path = './data'
  



exploration_filename = data_path+'/DDPG_exploration.csv'
test_filename = data_path+'/DDPG_test.csv'
agents_path = data_path+'/agents/DDPG/checkpoints/LeNet_DDPG_{}'.format(log_name)

current_state = 'layer_weights'
next_state = 'layer_weights'

epsilon_start_value = 0.9
epsilon_decay = 0.997
min_epsilon = 0.1
replay_buffer_size = 5000

rl_iterations = 2000
n_samples_mode = -1
strategy = None
tuning_batch_size = 128
tuning_epochs = 30
rl_batch_size = 128
verbose = 0
tuning_mode = 'layer'

layer_name_list = ['conv2d_1',  'dense', 'dense_1']

def dataset_preprocessing(img, label):
    img = tf.cast(img, tf.float32)
    img = img/255.0
    return img, label

def load_dataset(dataset_name, batch_size=128):
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

def create_model(dataset_name, train_ds, valid_ds):
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

def make_env(create_model, train_ds, valid_ds, test_ds, input_shape, layer_name_list, num_feature_maps, tuning_batch_size, tuning_epochs, verbose=0, tuning_mode='final', current_state_source='layer_input', next_state_source='layer_output', strategy=None, model_path='./data'):

    w_comprs = ['InsertDenseSVD'] 
    l_comprs = ['MLPCompression']
    compressors_list = w_comprs +  l_comprs

    parameters = {}
    parameters['InsertDenseSVD'] = {'layer_name': None, 'percentage': None}
    parameters['MLPCompression'] = {'layer_name': None, 'percentage': None}

    env = ModelCompressionSVDEnv(compressors_list, create_model, parameters, train_ds, valid_ds, test_ds, layer_name_list, input_shape, tuning_batch_size=tuning_batch_size, tuning_epochs=tuning_epochs, tuning_mode=tuning_mode, current_state_source=current_state_source, next_state_source=next_state_source, num_feature_maps=num_feature_maps, verbose=verbose,strategy=strategy, model_path=model_path)

    return env

def create_environments(dataset_names):
    environments = []
    for dataset in dataset_names:
        train_ds, valid_ds, test_ds, input_shape, _ = load_dataset(dataset, tuning_batch_size)
        create_model_dataset = partial(create_model, dataset_name=dataset, train_ds=train_ds, valid_ds=valid_ds)
        new_create_model_func = partial(create_model_dataset, dataset_name=dataset, train_ds=train_ds, valid_ds=valid_ds)

        env = make_env(
                create_model=new_create_model_func, 
                train_ds=train_ds, 
                valid_ds=valid_ds, 
                test_ds=test_ds, 
                input_shape=input_shape, 
                layer_name_list=layer_name_list, 
                num_feature_maps=n_samples_mode,
                tuning_batch_size=tuning_batch_size,
                tuning_epochs = tuning_epochs, 
                verbose=verbose, 
                tuning_mode=tuning_mode, 
                current_state_source=current_state, 
                next_state_source=next_state, 
                strategy=strategy, 
                model_path=data_path)

        environments.append(env)

    return environments

envs = create_environments(dataset_names)
envs[0].model.summary()

conv_shape, dense_shape = envs[0].observation_space()
action_space = envs[0].action_space()
num_actions = 1

fc_n_actions = conv_n_actions = num_actions
logger.info('Biggest convolution shape is {}. Biggest dense shape is {}.'.format(conv_shape, dense_shape))

upper_bound = action_space['high']
lower_bound = action_space['low']

logger.info("Max Value of Action ->  {}".format(upper_bound))
logger.info("Min Value of Action ->  {}".format(lower_bound))


agent = DDPGWeights2D("ddpg_agent_conv", conv_shape,
                        fc_n_actions, epsilon=epsilon_start_value, layer_type='conv')
target_agent = DDPGWeights2D(
    "target_network_fc", conv_shape,
                        fc_n_actions, epsilon=epsilon_start_value, layer_type='conv')

agent.actor.summary()
agent.critic.summary()



try:
    agent.actor.load_weights(agents_path+'fc_actor')
    agent.critic.load_weights(agents_path+'fc_critic')
    target_agent.actor.load_weights(agents_path+'fc_actor')
    target_agent.critic.load_weights(agents_path+'fc_critic')
except:
    logger.info('Failed to find pretrained models for the RL agents.')
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
def update_agent(
    state_batch,
    action_batch,
    reward_batch,
    next_state_batch,
):
    # Training and updating Actor & Critic networks.
    # See Pseudo Code.
    with tf.GradientTape() as tape:
        target_actions = target_agent.actor(next_state_batch, training=True)
        y = reward_batch + gamma * target_agent.critic([next_state_batch, target_actions], training=True
        )
        critic_value = agent.critic([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, agent.critic.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, agent.critic.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = agent.actor(state_batch, training=True)
        critic_value = agent.critic([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, agent.actor.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, agent.actor.trainable_variables)
    )

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
replay_buffer = ReplayBufferMultipleDatasets(replay_buffer_size, dataset_names)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

def play_and_record(env, run_id, test_number, dataset_name, save_name, n_games=10, exploration=True):
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
        # data = []
        for k in range(1, len(env.layer_name_list)+1):
            tf.keras.backend.clear_session()
            # Get the current layer name
            current_layer_name = env.layer_name_list[env._layer_counter]

            # Action is the mode of the action.
            action = agent.policy(s, exploration=exploration)
            logger.debug(f'Action for layer {current_layer_name} layer is {action}')

            # Apply action
            new_s, r, done, info = env.step(action)

            num_inst = s.shape[0]
            replay_buffer.add_multiple(s, [action]*num_inst, [info['reward_step']]*num_inst, new_s, [None]*num_inst, dataset_name)

            batch = sample_batch(replay_buffer, rl_batch_size)
            update_agent(**batch)

            update_target(target_agent.actor.variables, agent.actor.variables, tau)
            update_target(target_agent.critic.variables, agent.critic.variables, tau)
            agent.actor.save_weights(agents_path+'fc_actor')
            agent.critic.save_weights(agents_path+'fc_critic')

            logger.debug(f'Iteration {it} - Layer {current_layer_name} {k}/{len(env.layer_name_list)}\tChosen action {action} has {r} reward.')
            logger.debug(info)

            row = {'state': s, 'action': action, 'reward': r,
                   'next_state': new_s, 'done': done, 'info': info}


            # data.append(row)
                

            s = env.get_state('current_state')

            if done:
                s = env.reset()
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
        # for row in data:
        #     logging.debug('Storing instance in fc replay.')
        #     replay_buffer.add_multiple(row['state'], [row['action']],[rewards[-1]], row['next_state'], ['None'])
        #     logging.debug('Sampling instances from fc replay.')
        # for row in data:
        #     logging.debug('Storing instance in fc replay.')
        #     replay_buffer.add_multiple(row['state'], [row['action']],[rewards[-1]], row['next_state'], ['None'])
        #     logging.debug('Sampling instances from fc replay.')

    logger.info(f'Evaluation of {n_games} took {total_time} secs. An average of {total_time/n_games} secs per game.')

    return np.mean(rewards), np.mean(acc), np.mean(weights)

num_datasets = len(dataset_names)

num_tests = (rl_iterations//10) + 1

weights_history_tests = np.zeros(shape=(num_tests, num_datasets))
acc_history_tests = np.zeros(shape=(num_tests, num_datasets))
rw_history_tests = np.zeros(shape=(num_tests, num_datasets))

for idx, env in enumerate(envs):
    weights_history_tests[0, idx ] = env.weights_before
    acc_history_tests[0, idx] = env.test_acc_before

# for idx, env in enumerate(envs):
#     dataset_name = dataset_names[idx]
#     rw, acc, weights = play_and_record(env, run_id=run_id, test_number=-1, dataset_name=dataset_name,save_name=exploration_filename,n_games=10, exploration=True)
       

test_counter = 1
with tqdm(total=rl_iterations,
        bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Last 3 RW: {postfix[0][0]:.2f}, {postfix[0][1]:.2f} & {postfix[0][2]:.2f} W: {postfix[1][0]:.2f}, {postfix[1][1]:.2f} & {postfix[1][2]:.2f} Acc: {postfix[2][0]:.2f}, {postfix[2][1]:.2f} & {postfix[2][2]:.2f}] Replay: {postfix[3]} .",
        postfix=[
            dict({0: 0, 1: 0, 2: np.mean(rw_history_tests[0])}),
            dict({0: 0, 1: 0, 2: np.mean(acc_history_tests[0])}),
            dict({0: 0, 1: 0, 2: np.mean(weights_history_tests[0])}), 
            len(replay_buffer)]) as t:
    for i in range(1, rl_iterations+1):
        logger.info(f'Iteration {i} of {rl_iterations}.')
        for idx, env in enumerate(envs):
            dataset_name = dataset_names[idx]
            logger.info(f'Generating samples for dataset {dataset_name}.')
            rw, acc, weights = play_and_record(env, run_id=run_id, test_number=i, dataset_name=dataset_name,save_name=exploration_filename,n_games=1, exploration=True)
            logger.info(f'Training stats for dataset {dataset_name} were rw:{rw}, acc:{acc}, w:{weights}.')
        if i %10==0: 
            logger.info(f'Testing datasets.')
            for idx, env in enumerate(envs):
                dataset_name = dataset_names[idx]
                logger.info(f'Testing agent for dataset {dataset_name}.')
                rw, acc, weights = play_and_record(env, run_id=run_id, test_number=i, dataset_name=dataset_name,save_name=test_filename,n_games=1, exploration=False)
                logger.info(f'Testing stats for dataset {dataset_name} were rw:{rw}, acc:{acc}, w:{weights}.')
                rw_history_tests[test_counter, idx] = rw
                acc_history_tests[test_counter, idx] = acc
                weights_history_tests[test_counter, idx] = weights

            


            t.postfix[0][2] = np.mean(rw_history_tests[test_counter])
            
            
            try:
                t.postfix[0][1] = np.mean(rw_history_tests[test_counter-1])
            except IndexError:
                t.postfix[0][1] = 0
            try:
                t.postfix[0][0] =  np.mean(rw_history_tests[test_counter-2])
            except IndexError:
                t.postfix[0][0] = 0

            t.postfix[1][2] = np.mean(weights_history_tests[test_counter])
            try:
                t.postfix[1][1] = np.mean(weights_history_tests[test_counter-1])
            except IndexError:
                t.postfix[1][1] = 0
            try:
                t.postfix[1][0] = np.mean(weights_history_tests[test_counter-2])
            except IndexError:
                t.postfix[1][0] = 0

            t.postfix[2][2] = np.mean(acc_history_tests[test_counter])
            try:
                t.postfix[2][1] = np.mean(acc_history_tests[test_counter-1])
            except IndexError:
                t.postfix[2][1] = 0
            try:
                t.postfix[2][0] = np.mean(acc_history_tests[test_counter-2])
            except IndexError:
                t.postfix[2][0] = 0


            test_counter += 1
            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            ax1.title.set_text('Accuracy')
            for idx, dataset_name in enumerate(dataset_names):
                ax1.plot(acc_history_tests[:test_counter, idx])
            ax1.legend(dataset_names)
            ax2.title.set_text('Weights')
            for idx, dataset_name in enumerate(dataset_names):
                ax2.plot(weights_history_tests[:test_counter, idx])
            ax2.legend(dataset_names)
            ax3.title.set_text('Reward')
            for idx, dataset_name in enumerate(dataset_names):
                ax3.plot(rw_history_tests[:test_counter, idx])
            ax3.legend(dataset_names)
            plt.xlabel('Epochs')
            plt.savefig(f'./data/figures/DDPG_test_stats.png')
            plt.close()

        t.postfix[3] = len(replay_buffer)
        t.update()