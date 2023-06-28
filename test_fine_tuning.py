import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_datasets as tfds
import logging

from CompressionLibrary.agent_evaluators import make_env_adadeep

from uuid import uuid4
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
import gc


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



dataset_names = ['fashion_mnist', 'kmnist', 'mnist']


tuning_epochs = 90
eval_n_samples = 10
batch_size_per_replica = 256


file_name = data_path + f'/stats/Test_fine_tuning_final_{tuning_epochs}.csv'
verbose = 0

# Only needed by env. Not used by agent.
n_samples_mode = 256
current_state = 'layer_input'
next_state = 'layer_output'


layer_name_list = ['conv2d_1',  'dense', 'dense_1']
tuning_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync


logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(f'/home/A00806415/DCC/ModelCompression/data/Fine_tuning_{tuning_epochs}.log', 'w+')],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

log = logging.getLogger('tensorflow')
log.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

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

    train_ds = train_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = validation_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, test_ds, input_shape, num_classes

def calculate_reward(stats: dict) -> float:
   return 1 - (stats['weights_after']/stats['weights_before']) + stats['accuracy_after'] - 0.9 * stats['accuracy_before']

input_shape = (28,28,1)

def create_environments(dataset_names):
    environments = []
    for dataset in dataset_names:
        train_ds, valid_ds, test_ds, input_shape, _ = load_dataset(dataset, tuning_batch_size)
        new_func = partial(create_model, dataset_name=dataset, train_ds=train_ds, valid_ds=valid_ds)
        env = make_env_adadeep(
            create_model=new_func, 
            reward_func=calculate_reward,
            train_ds=train_ds, 
            valid_ds=valid_ds, 
            test_ds=test_ds, 
            input_shape=input_shape, 
            layer_name_list=layer_name_list, 
            num_feature_maps=n_samples_mode, 
            tuning_batch_size=tuning_batch_size, 
            tuning_epochs=tuning_epochs, 
            verbose=verbose, 
            tuning_mode='final', 
            current_state_source=current_state, 
            next_state_source=next_state, 
            strategy=strategy)

        environments.append(env)

    return environments


envs = create_environments(dataset_names)
envs[0].model.summary()

def play_and_record(actions, env, run_id, test_number, dataset_name, save_name, n_games=1):
    # initial state
    s = env.reset()
    # Play the game for n_steps as per instructions above

    logger = logging.getLogger(__name__)
    rewards = []
    try:
        df_results = pd.read_csv(save_name)
    except:
        df_results = pd.DataFrame()

    for it in range(n_games):
        start = datetime.now()
        s = env.reset()
        for k in range(1, len(env.layer_name_list)+1):
            tf.keras.backend.clear_session()
            # Get the current layer name
            current_layer_name = env.layer_name_list[env._layer_counter]
            # Get the layer.
            action = actions[k-1]

            logger.debug(f'Compressing layer {current_layer_name} using action {action}.')

            # Apply action
            new_s, r, done, info = env.step(action)

            logger.debug(f'Iteration {it} - Layer {current_layer_name} {k}/{len(env.layer_name_list)}\tChosen action {action} has {r} reward.')
            logger.debug(info)

            row = {'state': s, 'action': action, 'reward': r,
                   'next_state': new_s, 'done': done, 'info': info}

            s = env.get_state('current_state')

            if done:
                s = env.reset()
                break
            gc.collect()

        info['actions'] = ','.join(info['actions'])
        info['run_id'] = run_id
        info['test_number'] = test_number
        info['game_id'] = it
        info['dataset'] = dataset_name
        del info['layer_name']
        new_row = pd.DataFrame(info, index=[0])
        df_results = pd.concat([df_results, new_row], ignore_index=True)

        # Correct reward is the last value of r.
        rewards.append(r)
        end = datetime.now()
        time_diff = (end - start).total_seconds()
        logger.info(f'Took {time_diff} seconds for one compression.')
      

    df_results.to_csv(save_name, index=False)

    return np.mean(rewards)


class Agent:
    def __init__(self, conv_n_actions, fc_n_actions):
        """Base class for an agent."""
        self.logger = logging.getLogger(__name__)

        self.sequences = [[conv1, fc1, fc2] for conv1 in range(conv_n_actions) for fc1 in range(fc_n_actions-1) for fc2 in range(fc_n_actions-1)]
        for conv1 in range(conv_n_actions):
            self.sequences.append([conv1, 4, 0])
        print(self.sequences)

    def policy(self, seq):
        actions = self.sequences[seq]
        return actions
        

conv_n_actions, fc_n_actions = envs[0].action_space()


agent = Agent(conv_n_actions, fc_n_actions)

n_sequences = conv_n_actions * (fc_n_actions-1) * (fc_n_actions-1) + (conv_n_actions)
n_datasets = len(dataset_names)

num_iterations = n_datasets * n_sequences

with tqdm(total=num_iterations) as pbar:
    for env_idx, env in enumerate(envs):
        for seq in range(n_sequences):
            dataset_name = dataset_names[env_idx]
            actions = agent.policy(seq)
            logger.debug(f'Dataset: {dataset_name}. Test actions: {actions}. ')
            play_and_record(actions, env, run_id=run_id, test_number=1, dataset_name=dataset_name, save_name=file_name, n_games=eval_n_samples)
            pbar.update(1)