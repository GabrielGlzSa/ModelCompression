
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_datasets as tfds

import logging
import random

import tensorflow.keras as keras
from CompressionLibrary.utils import calculate_model_weights
from CompressionLibrary.reward_functions import reward_MnasNet as calculate_reward
from uuid import uuid4
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from CompressionLibrary.CompressionTechniques import InsertDenseSVD, MLPCompression
import copy

import gc

current_os = 'windows'


dataset_name = 'kmnist'
batch_size = 64

agent_name = 'simulated_annealing' + '_' + dataset_name

if current_os == 'windows':
    data_path = f'G:\\Python projects\\ModelCompressionRL\\data\\'
    log_path = f'G:\\Python projects\\ModelCompressionRL\\data\\logs\\ModelCompression_{agent_name}.log'
    exploration_filename = data_path + f'stats/{agent_name}_training.csv'
    test_filename = data_path + f'stats\\{agent_name}_testing.csv'
    figures_path = data_path+f'figures\\{agent_name}'
    datasets_path = data_path+"datasets"
else:
    data_path = './data'
    log_path = f'/home/A00806415/DCC/ModelCompression/data/logs/ModelCompression_{agent_name}.log'
    exploration_filename = data_path + f'/stats/{agent_name}_training.csv'
    test_filename = data_path + f'/stats/{agent_name}_testing.csv'
    figures_path = data_path+f'/figures/{agent_name}'


logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(log_path, 'w+')],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

log = logging.getLogger('tensorflow')
log.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


layer_name_list = ['conv2d_1',  'dense', 'dense_1']

def create_model(dataset_name, train_ds, valid_ds):
    if current_os =='windows':
        checkpoint_path = f"G:\\Python projects\\ModelCompressionRL\\data\\models\\lenet_{dataset_name}\\cp.ckpt"
    else:
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
                                split=['train[:80%]', 'train[80%:]','test'], data_dir=datasets_path)

    (train_examples, validation_examples, test_examples) = splits
    num_examples = info.splits['train'].num_examples

    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape

    input_shape = (28,28,1)

    train_ds = train_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = validation_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, test_ds, input_shape, num_classes


def get_max_hidden_units(model, layer_list):
    max_values = []
    for layer_name in layer_list:
        layer = model.get_layer(layer_name)
        if isinstance(layer, tf.keras.layers.Conv2D):
            kernel, bias = layer.get_weights()
            h, w, c, filters = kernel.shape
            weights = tf.reshape(kernel, shape=[-1, filters])
            input_size, _ = weights.shape
            units = filters
            
        elif isinstance(layer, tf.keras.layers.Dense):
            weights, bias = layer.get_weights()
            input_size , units = weights.shape
            
        max_hidden_units = (input_size * units)//(input_size+units)
        max_values.append(max_hidden_units)

    return max_values



train_ds, valid_ds, test_ds, input_shape, _ = load_dataset(dataset_name, batch_size)

parameters = {}
parameters['InsertDenseSVD'] = {'layer_name': None, 'percentage': None, 'hidden_units':None}
parameters['MLPCompression'] = {'layer_name': None, 'percentage': None, 'hidden_units':None}


optimizer = tf.keras.optimizers.Adam(1e-5)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
verbose = False

# Create a the original model to calculate stats before.
temp_model = create_model(dataset_name=dataset_name, train_ds=train_ds, valid_ds=valid_ds)
weights_before = calculate_model_weights(temp_model)
test_loss, test_acc_before = temp_model.evaluate(test_ds, verbose=verbose)
val_loss, val_acc_before = temp_model.evaluate(valid_ds, verbose=verbose)


max_hidden_units = get_max_hidden_units(temp_model, layer_name_list)
del temp_model

logger.info(f'Max number of singular values per layer are : {max_hidden_units}.')
def evaluation_function(ind):
    callbacks = []
    model = create_model(dataset_name=dataset_name, train_ds=train_ds, valid_ds=valid_ds)
    for action_idx, layer_name in enumerate(layer_name_list):
        layer = model.get_layer(layer_name)
        action = ind[action_idx]
        logger.debug(f'Using {action} singular values for layer {layer_name}. Max number is {max_hidden_units[action_idx]}.')
        if action < max_hidden_units[action_idx]:
            if isinstance(layer, tf.keras.layers.Conv2D):
                compressor = MLPCompression(model=model, dataset=train_ds, optimizer=optimizer, loss=loss_object, metrics=train_metric,
                            fine_tuning=False, input_shape=input_shape)
                compressor_name = 'MLPCompression'
            elif isinstance(layer, tf.keras.layers.Dense):
                compressor = InsertDenseSVD(model=model, dataset=train_ds, optimizer=optimizer, loss=loss_object, metrics=train_metric,
                            fine_tuning=False, input_shape=input_shape)
                compressor_name = 'InsertDenseSVD'
            

            compressor.callbacks = callbacks

            parameters[compressor_name]['layer_name'] = layer_name
            parameters[compressor_name]['hidden_units'] = action

            compressor.compress_layer(**parameters[compressor_name])

            # Get compressed model
            model = compressor.get_model()
            callbacks = compressor.callbacks
            

    
    test_loss, test_acc_after = model.evaluate(test_ds, verbose=verbose)
    val_loss, val_acc_after = model.evaluate(valid_ds, verbose=verbose)
    weights_after = calculate_model_weights(model)
    stats = {
                'weights_before': weights_before, 
                'weights_after': weights_after, 
                'accuracy_after': test_acc_after, 
                'accuracy_before': test_acc_before}

    reward = calculate_reward(stats)
    logger.debug(f'Solution {ind} has {weights_after} weights and an accuracy of {test_acc_after}. Reward is {reward}')
    return reward, stats


def mutation(ind, temperature, max_delta=10):
    new_ind = []
    max_value = max_delta 
    for action_idx, action in enumerate(ind):
        delta = np.random.randint(low=-max_value, high=max_value)
        new_action = np.clip(action + delta, a_min=1, a_max=max_hidden_units[action_idx]+1)
        new_ind.append(new_action)
    return new_ind


# function used to determine which evaluation is better (> or <)
better = np.greater_equal
best_stats = None
max_temp = 10

def markov_chain(solution, fu, temperature, chain_length):
    global best_stats
    accepted_transitions = 0
    best_solution = copy.deepcopy(solution)
    fbest = fu
    for i in range(chain_length):
        neighbor = mutation(solution, temperature)
        fneighbor, stats = evaluation_function(neighbor)
        logger.debug(f'Iteration {i+1}/{chain_length} - u:{solution} fu:{fu} \t\tv:{neighbor} fv:{fneighbor}')
        if better(fneighbor, fu):
            logger.debug('Accepted because fv is better than fu.')
            solution = neighbor
            fu = fneighbor
            accepted_transitions += 1

            if better(fneighbor, fbest):
                best_solution = copy.deepcopy(neighbor)
                fbest = fneighbor
                best_stats = stats
        else:
            abs_delta = np.abs(fneighbor - fu)
            r = random.random()
            p = np.exp(-abs_delta / temperature)
            logger.debug(f' Accept worse {r} < {p}?')
            if r < p:
                logger.debug(f'Accepted')
                solution = neighbor
                fu = fneighbor
                accepted_transitions += 1
            else:
                logger.debug(f'Rejected')
                

    return solution, fu, accepted_transitions / chain_length, best_solution, fbest

def fix_solution(ind):
    new_ind = []
    for action_idx, action in enumerate(ind):
        new_ind.append(np.clip(action, 1, max_hidden_units[action_idx]))
    return new_ind



def simulated_annealing(iterations=1, alpha=0.8, beta=1.1, temperature=0.1, chain_length=10, Rmin=0.9, init_temp=False):
    global best_stats
    u = np.random.randint(1, np.max(max_hidden_units) + 1, size=len(max_hidden_units))
    u = fix_solution(u)
    fu, best_stats = evaluation_function(u)
    Ra = 0
    best_sol = copy.copy(u)
    fbest = fu
    if init_temp:
        while Ra < Rmin:
            u, Ra, best_sol, fbest = markov_chain(u, temperature, chain_length)
            if better(fbest,new_fbest):
                best_sol = new_best_sol
                fbest = new_fbest
            T = T * beta
            logger.debug(f'Temperature is {temperature}. Percentage of successful transitions for {chain_length}: {Ra}.')


        u = np.random.randint(1, np.max(max_hidden_units) + 1, size=len(max_hidden_units))
        u = fix_solution(u)
    adaptative_cooling = lambda fb, fu: alpha if  better(fu, fb) else beta
    with tqdm(total=iterations, bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, RW: {postfix[0]:.3f} W: {postfix[1]:.0f} Acc: {postfix[2]:.3f}] Temperature={postfix[3]:.3f}. Ra={postfix[4]}.",
        postfix=[fbest, best_stats['weights_after'], best_stats['accuracy_after'], temperature, Ra ]) as t:
        for _ in range(iterations):
            u, fu, Ra, new_best_sol, new_fbest = markov_chain(u, fu, temperature, chain_length)
            if better(new_fbest,fbest):
                best_sol = new_best_sol
                fbest = new_fbest
            

            temperature = min(adaptative_cooling(fbest, new_fbest) * temperature, max_temp)

            logger.debug(f'Temperature is {temperature}. Percentage of successful transitions for {chain_length}: {Ra}.')
            logger.debug(f'Best solution: {best_sol}. F(best)={fbest}')

            t.postfix[0] = fbest
            t.postfix[1] = best_stats['weights_after']
            t.postfix[2] = best_stats['accuracy_after']
            t.postfix[3] = temperature
            t.postfix[4] = Ra
            t.update()


        



simulated_annealing(iterations=100, temperature=1, alpha=0.9, beta=1.5, chain_length=50, Rmin=0.9)