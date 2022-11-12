
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_datasets as tfds
import logging

from CompressionLibrary.agent_evaluators import make_env_imagenet, evaluate_agents, play_and_record
from CompressionLibrary.reinforcement_models import DuelingDQNAgent
from CompressionLibrary.replay_buffer import ReplayBuffer

from uuid import uuid4
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import partial


dataset_names = ['mnist', 'fashion_mnist']

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
  
if strategy:
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

log_name = '-'.join(dataset_names)
logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(f'/home/A00806415/DCC/ModelCompression/data/ModelCompression_DuelingDQN_{log_name}.log', 'w+')],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

log = logging.getLogger('tensorflow')
log.setLevel(logging.ERROR)




current_state = 'layer_input'
next_state = 'layer_output'

epsilon_start_value = 0.9
epsilon_decay = 0.995
min_epsilon = 0.1
replay_buffer_size = 10 ** 6

verbose = 0
rl_iterations = 1000
eval_n_samples = 5
n_samples_mode = 256
batch_size_per_replica = 128
tuning_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
rl_batch_size = tuning_batch_size
tuning_epochs = 30

layer_name_list = ['conv2d_1',  'dense', 'dense_1']

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


input_shape = (28,28,1)

def create_environments(dataset_names):
    environments = []
    for dataset in dataset_names:
        train_ds, valid_ds, test_ds, input_shape, _ = load_dataset(dataset, tuning_batch_size)
        new_func = partial(create_model, dataset_name=dataset, train_ds=train_ds, valid_ds=valid_ds)
        env = make_env_imagenet(
            create_model=new_func, 
            train_ds=train_ds, 
            valid_ds=valid_ds, 
            test_ds=test_ds, 
            input_shape=input_shape, 
            layer_name_list=layer_name_list, 
            num_feature_maps=n_samples_mode, 
            tuning_batch_size=tuning_batch_size, 
            tuning_epochs=tuning_epochs, 
            verbose=verbose, 
            tuning_mode='layer', 
            current_state_source=current_state, 
            next_state_source=next_state, 
            strategy=strategy, 
            model_path=data_path)

        environments.append(env)

    return environments


envs = create_environments(dataset_names)
envs[0].model.summary()

conv_shape, dense_shape = envs[0].observation_space()
conv_n_actions, fc_n_actions = envs[0].action_space()

print(conv_shape, dense_shape)

def load_weigths_into_target_network(agent, target_network):
    """ assign target_network.weights variables to their respective agent.weights values. """
    for i in range(len(agent.model.layers)):
        target_network.model.layers[i].set_weights(
            agent.model.layers[i].get_weights())


with strategy.scope():
    
    fc_agent = DuelingDQNAgent("dqn_agent_fc", dense_shape,
                        fc_n_actions, epsilon=epsilon_start_value, layer_type='fc')
    fc_target_network = DuelingDQNAgent(
        "target_network_fc", dense_shape, fc_n_actions, layer_type='fc')

    fc_agent.model.summary()

    conv_agent = DuelingDQNAgent("dqn_agent_conv", conv_shape,
                        conv_n_actions, epsilon=epsilon_start_value, layer_type='cnn')
    conv_target_network = DuelingDQNAgent(
        "target_network_conv", conv_shape, conv_n_actions, layer_type='cnn')


    try:
        conv_target_network.model.load_weights(
            data_path+'/checkpoints/DuelingDQN_{}_my_checkpoint_conv_agent'.format(log_name))
        fc_target_network.model.load_weights(
            data_path+'/checkpoints/DuelingDQN_{}_my_checkpoint_fc_agent'.format(log_name))
    except:
        print('Failed to find pretrained models for the RL agents.')
        pass

    load_weigths_into_target_network(fc_agent, fc_target_network)
    load_weigths_into_target_network(conv_agent, conv_target_network)

    for w, w2 in zip(fc_agent.model.trainable_weights, fc_target_network.model.trainable_weights):
        tf.assert_equal(w, w2)
    for w, w2 in zip(conv_agent.model.trainable_weights, conv_target_network.model.trainable_weights):
        tf.assert_equal(w, w2)

    print("It works!")

    optimizer_conv = tf.keras.optimizers.Adam(1e-5)
    optimizer_fc = tf.keras.optimizers.Adam(1e-5)



def training_loop(state, action, rewards, next_state, done, agent, target_agent, loss_optimizer, n_actions, gamma=0.01):
    agent_best_actions = tf.cast(tf.math.argmax(agent.get_qvalues(next_state), axis=1), tf.int32)
    indices = tf.stack([tf.range(state.shape[0]), agent_best_actions], axis=1)
    target_agent_qvalues = tf.gather_nd(target_agent.get_qvalues(next_state), indices=indices)
    reference_qvalues = rewards + gamma * target_agent_qvalues * (1.0 - done)

    masks = tf.one_hot(action, n_actions)
    with tf.GradientTape() as tape:
        q_values = agent.get_qvalues(state)
        selected_action_qvalues = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        td_loss = tf.reduce_mean((reference_qvalues - selected_action_qvalues) ** 2)

    gradients = tape.gradient(td_loss, agent.model.trainable_weights)
    loss_optimizer.apply_gradients(zip(gradients, agent.model.trainable_weights))
    return td_loss



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
        'state': tf.cast(obs_batch.to_tensor(), tf.float32),
        'action': tf.cast(act_batch, tf.int32),
        'rewards': tf.cast(reward_batch, tf.float32),
        'next_state': tf.cast(next_obs_batch.to_tensor(), tf.float32),
        'done': tf.cast(is_done_batch, tf.float32),
    }
    

fc_exp_replay = ReplayBuffer(replay_buffer_size)
conv_exp_replay = ReplayBuffer(replay_buffer_size)

print('There are {} conv and {} fc instances.'.format(len(conv_exp_replay), len(fc_exp_replay)))

for idx, env in enumerate(envs):
    dataset = dataset_names[idx]
    play_and_record(conv_agent, fc_agent, env, conv_exp_replay, fc_exp_replay, run_id=run_id, test_number=0-idx, dataset_name=dataset,save_name=data_path+'/DuelingDQN_training_exploration.csv', n_games=5)

print('There are {} conv and {} fc instances.'.format(len(conv_exp_replay), len(fc_exp_replay)))

with tqdm(total=rl_iterations,
        bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Epsilon: {postfix[0]:.4f} Last 3 RW: {postfix[1][0]:.2f}, {postfix[1][1]:.2f} & {postfix[1][2]:.2f} W: {postfix[2][0]:.2f}, {postfix[2][1]:.2f} & {postfix[2][2]:.2f} Acc: {postfix[3][0]:.2f}, {postfix[3][1]:.2f} & {postfix[3][2]:.2f}] Replay: conv:{postfix[4]}/fc:{postfix[5]}.",
        postfix=[conv_agent.epsilon, dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0}), len(conv_exp_replay), len(fc_exp_replay)]) as t:
    for i in range(rl_iterations):

        # generate new sample
        for idx, env in enumerate(envs):
            dataset = dataset_names[idx]
            play_and_record(conv_agent, fc_agent, env, conv_exp_replay, fc_exp_replay, run_id=run_id, test_number=i, dataset_name=dataset,save_name=data_path+'/DuelingDQN_training_exploration.csv',n_games=1)

        # train fc
        batch_data = sample_batch(conv_exp_replay, batch_size=rl_batch_size)
        batch_data['agent'] = conv_agent
        batch_data['target_agent'] = conv_target_network
        batch_data['loss_optimizer'] = optimizer_conv
        batch_data['n_actions'] = conv_n_actions
        conv_loss_t = training_loop(**batch_data)
        td_loss_history_conv.append(conv_loss_t)

        # train
        batch_data = sample_batch(fc_exp_replay, batch_size=rl_batch_size)
        batch_data['agent'] = fc_agent
        batch_data['target_agent'] = fc_target_network
        batch_data['loss_optimizer'] = optimizer_fc
        batch_data['n_actions'] = fc_n_actions
        fc_loss_t = training_loop(**batch_data)
        td_loss_history_fc.append(fc_loss_t)

        
        conv_agent.epsilon = max(conv_agent.epsilon * epsilon_decay, min_epsilon)
        fc_agent.epsilon = max(fc_agent.epsilon * epsilon_decay, min_epsilon)
        t.postfix[0] = conv_agent.epsilon
        t.postfix[5]= len(fc_exp_replay)
        t.postfix[4]= len(conv_exp_replay)
        
        # adjust agent parameters
        if i % 10 == 0:
            load_weigths_into_target_network(conv_agent, conv_target_network)
            conv_target_network.model.save_weights(
                data_path+'/checkpoints/DuelingDQN_{}_my_checkpoint_conv_agent'.format(log_name))
            
            

            load_weigths_into_target_network(fc_agent, fc_target_network)
            fc_target_network.model.save_weights(
                    data_path+'/checkpoints/DuelingDQN_{}_my_checkpoint_fc_agent'.format(log_name))
            
            temp_rw = []
            temp_acc = []
            temp_w = []
            for idx, env in enumerate(envs):
                dataset = dataset_names[idx]
                rw, acc, weights = evaluate_agents(env, conv_agent, fc_agent,run_id=run_id,test_number=i//10, dataset_name=dataset,save_name=data_path+'/DuelingDQN_test_evaluate.csv', n_games=eval_n_samples)            
                temp_rw.append(rw)
                temp_acc.append(acc)
                temp_w.append(weights)

            rw = np.mean(temp_rw)
            acc = np.mean(temp_acc)
            weights = np.mean(temp_w)

            mean_rw_history.append(rw)
            mean_acc_history.append(acc)
            mean_weights_history.append(weights)

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

            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            ax1.title.set_text('Accuracy')
            ax1.plot(mean_acc_history)
            ax2.title.set_text('Weights(%)')
            ax2.plot(mean_weights_history)
            ax3.title.set_text('Reward')
            ax3.plot(mean_rw_history)
            plt.savefig('./data/figures/DuelingDQN_{}_test_stats.png'.format(log_name))
            plt.close()

        t.update()
        
        plt.plot(td_loss_history_conv, color='r')
        plt.plot(td_loss_history_fc, color='b')
        plt.savefig('./data/figures/DuelingDQN_{}_td_loss.png'.format(log_name))
        plt.close()