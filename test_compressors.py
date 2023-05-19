import tensorflow as tf
import tensorflow_datasets as tfds
import logging
from uuid import uuid4
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from CompressionLibrary.agent_evaluators import make_env_adadeep, evaluate_agents
from CompressionLibrary.reinforcement_models import DuelingDQNAgent


agents_names = list(map(lambda x: 'LeNet_DDDQN_MKII_'+x, ['fashion_mnist','kmnist', 'mnist', 'fashion_mnist-kmnist', 'fashion_mnist-mnist','kmnist-mnist']))
agents_names = agents_names + ['LeNet_DDDQN_MKIII_generalist_fashion_mnist-kmnist-mnist']

dataset_names = ['fashion_mnist','kmnist', 'mnist']
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

strategy = None
data_path = './data/'
  
log_name = 'test_agents_validation_ds_all'
test_filename = data_path + 'stats/DDDQN_MKII_{}.csv'.format(log_name)
agents_path = data_path+'agents/DDDQN/checkpoints/'

if strategy:
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(data_path + f'logs/{log_name}.log', 'w+')],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

log = logging.getLogger('tensorflow')
log.setLevel(logging.ERROR)


current_state = 'layer_input'
next_state = 'layer_output'

epsilon_start_value = 0.9
verbose = 0
eval_n_samples = 1
n_samples_mode = -1
tuning_batch_size = 256
tuning_epochs = 30

layer_name_list = ['conv2d_1',  'dense', 'dense_1']


def calculate_reward(stats: dict) -> float:
   return 1 - (stats['weights_after']/stats['weights_before']) + stats['accuracy_after'] - 0.9 * stats['accuracy_before']


def create_model(dataset_name, train_ds, valid_ds):
    checkpoint_path = data_path+ f"models/lenet_{dataset_name}/cp.ckpt"
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
            tuning_mode='layer',
            get_state_from='validation',
            current_state_source=current_state, 
            next_state_source=next_state, 
            strategy=strategy, 
            model_path=data_path)

        environments.append(env)

    return environments

envs = create_environments(dataset_names)
envs[0].model.summary()
print(f'{len(envs)} envs')

conv_shape, dense_shape = envs[0].observation_space()
conv_n_actions, fc_n_actions = envs[0].action_space()
    
fc_agent = DuelingDQNAgent(name="ddqn_agent_fc", state_shape=dense_shape,
                    n_actions=fc_n_actions, epsilon=epsilon_start_value, layer_type='fc')

conv_agent = DuelingDQNAgent(
    name="ddqn_agent_conv", state_shape=conv_shape, n_actions=conv_n_actions, epsilon=epsilon_start_value, layer_type='cnn')

n_datasets = len(dataset_names)
n_agents = len(agents_names)
iterations =  n_datasets * n_agents 
print(iterations)
logger = logging.getLogger(__name__)

with tqdm(total=iterations) as t:
    for i in range(iterations):
        env_idx = i // n_agents
        agent_idx = i % n_agents
        agent_name = agents_names[agent_idx]
        dataset_name = dataset_names[env_idx]
        env = envs[env_idx]
        logger.debug(f'Dataset: {dataset_name}. Agent: {agent_name}.')
        conv_agent.model.load_weights(agents_path+agent_name+'_conv_agent.ckpt')
        fc_agent.model.load_weights(agents_path+agent_name+'_fc_agent.ckpt')

        rw, acc, weights = evaluate_agents(env, conv_agent, fc_agent,run_id=run_id,test_number=agent_name, dataset_name=dataset_name, agent_name=agent_name,save_name=test_filename, n_games=eval_n_samples)    