
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

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

from deap import base, creator
from deap import algorithms
import random
from deap import tools


current_os = 'linux'


dataset_name = 'imagenet2012'
batch_size = 32

agent_name = 'GA_100actions' + '_' + dataset_name


if current_os == 'windows':
    data_path = f'G:\\Python projects\\ModelCompressionRL\\data\\'
    log_path = f'G:\\Python projects\\ModelCompressionRL\\data\\logs\\ModelCompression_{agent_name}.log'
    exploration_filename = data_path + f'stats/{agent_name}_training.csv'
    test_filename = data_path + f'stats\\{agent_name}_testing.csv'
    figures_path = data_path+f'figures\\{agent_name}'
    datasets_path = "G:\\ImageNet 2012\\"#data_path+"datasets"
else:
    data_path = './data'
    log_path = f'/home/A00806415/DCC/ModelCompression/data/logs/ModelCompression_{agent_name}.log'
    exploration_filename = data_path + f'/stats/{agent_name}_training.csv'
    test_filename = data_path + f'/stats/{agent_name}_testing.csv'
    figures_path = data_path+f'/figures/{agent_name}'
    datasets_path = data_path 


# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(log_path, 'w+')],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

log = logging.getLogger('tensorflow')
log.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# layer_name_list = ['conv2d_1',  'dense', 'dense_1']
layer_name_list = [ 'block2_conv1', 'block2_conv2', 
                    'block3_conv1', 'block3_conv2', 'block3_conv3',
                    'block4_conv1', 'block4_conv2', 'block4_conv3',
                    'block5_conv1', 'block5_conv2', 'block5_conv3',
                    'fc1', 'fc2']


def create_model(dataset_name, train_ds, valid_ds):
    optimizer = tf.keras.optimizers.Adam(1e-5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()


    model = tf.keras.applications.vgg16.VGG16(
                            include_top=True,
                            weights='imagenet',
                            input_shape=(224,224,3),
                            classes=1000,
                            classifier_activation='softmax'
                        )

    model.compile(optimizer=optimizer, loss=loss_object,
                    metrics=train_metric)

    return model    

@tf.function
def imagenet_preprocessing(img, label):
    img = tf.cast(img, tf.float32)
    img = resize_image(img, (224,224))
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img, label

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

def load_dataset(dataset, batch_size=64):
    splits, info = tfds.load('imagenet2012', as_supervised=True, with_info=True, shuffle_files=True, 
                                split=['validation', 'validation','validation'], data_dir=datasets_path)

    #   splits, info = tfds.load('imagenet2012', as_supervised=True, with_info=True, shuffle_files=True, 
    #                               split=['train[:80%]', 'train[80%:]','validation'], data_dir=data_path)
                                
    (_, validation_examples, _) = splits
    num_examples = info.splits['validation'].num_examples#info.splits['train'].num_examples

    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape

    input_shape = (224,224,3)
    valid_ds = validation_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return valid_ds, input_shape, num_classes


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



valid_ds, input_shape, _ = load_dataset(dataset_name, batch_size)
train_ds = test_ds = valid_ds
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

logging.info(f'Max hidden values are {max_hidden_units}.')

logger.info(f'Max number of singular values per layer are : {max_hidden_units}.')

eval_dict = dict()
def evaluation_function(ind):
    ind = fix_solution(ind)
    callbacks = []
    model = create_model(dataset_name=dataset_name, train_ds=train_ds, valid_ds=valid_ds)
    logger.debug(f'Evaluating solution {ind}.')
    seq_key = ','.join(map(str,ind))
    logger.debug(f'Searching for {seq_key} in search dictionary.')
    if seq_key not in eval_dict.keys():
        logger.debug('Key not in dictionary. Evaluating solution.')
        for action_idx, layer_name in enumerate(layer_name_list):
            layer = model.get_layer(layer_name)
            action = ind[action_idx]
            logger.debug(f'Using {action}% of singular values for layer {layer_name}. 100% is {max_hidden_units[action_idx]}.')
            logger.debug(type(action))
            if action <= 100:
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
                parameters[compressor_name]['percentage'] = action

                compressor.compress_layer(**parameters[compressor_name])

                # Get compressed model
                model = compressor.get_model()
                callbacks = compressor.callbacks

        logger.debug('Evaluating model because it has not been explored before.')
        start = datetime.now()
        val_loss, val_acc_after = model.evaluate(valid_ds, verbose=verbose)
        test_loss, test_acc_after = val_loss, val_acc_after#model.evaluate(test_ds, verbose=verbose)
        
        end = datetime.now()
        logger.debug(f'Evaluation took {(end-start).total_seconds():2f} seconds. ')
    
    else:
        test_acc_after, weights_after = eval_dict[seq_key]
        logger.debug(f'Found evaluation of {ind}: ({test_acc_after}, {weights_after})')
        
    weights_after = calculate_model_weights(model)
    stats = {
                'weights_before': weights_before, 
                'weights_after': weights_after, 
                'accuracy_after': test_acc_after, 
                'accuracy_before': test_acc_before}

    reward = calculate_reward(stats)

    eval_dict[seq_key] = (stats['accuracy_after'], stats['weights_after'])

    logger.debug(f'Solution {ind} has {weights_after} weights and an accuracy of {test_acc_after}. Reward is {reward}')
    del model
    gc.collect()
    return stats['accuracy_after'], stats['weights_after']


def mutation(ind, indpb, max_delta=10):
    for action_idx in range(len(ind)):
        if random.uniform(0.0, 1.0) < indpb:
            delta = np.random.randint(low=-max_delta, high=max_delta)
            ind[action_idx] = int(np.clip(ind[action_idx] + delta, a_min=1, a_max=101))
    return ind,


def fix_solution(ind):
    for action_idx in range(len(ind)):
        ind[action_idx] = int(np.clip(ind[action_idx], 1, 101))
    return ind



creator.create("FitnessMaxMin", base.Fitness, weights=(1.0, -1.0,))
creator.create("Individual", list, fitness=creator.FitnessMaxMin)

toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, a=1, b=101)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=len(max_hidden_units))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutation, max_delta=10, indpb=0.1)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluation_function)


stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

pareto = tools.ParetoFront()
pop = toolbox.population(n=50)


pop[0] = creator.Individual(len(max_hidden_units) * [101])
pop[1] = creator.Individual(len(max_hidden_units) * [100])
pop[2] = creator.Individual(len(max_hidden_units) * [99])
pop[2] = creator.Individual(len(max_hidden_units) * [90])

start = datetime.now()
pop, logbook = algorithms.eaMuPlusLambda(population=pop,
                toolbox=toolbox,
                mu=26,
                lambda_= 26,
                cxpb=0.5,
                mutpb=0.5,
                halloffame=pareto,
                stats=stats,
                ngen=100,
                verbose=True)


end  = datetime.now()

table = []
logger.info(f'Took {(end - start).total_seconds()} seconds to run the GA algorithm.')
for ind in pareto:
    acc, weights = ind.fitness.values
    print(ind, acc, weights, 100*weights/weights_before)
    table.append([ind, val_acc_before, acc, weights_before, weights,100*weights/weights_before])
    logger.info(f"Solution {ind} has an accuracy of {acc:4f} and {int(weights)} weights ({100* weights/weights_before:3f}%).")


df = pd.DataFrame(table, columns=['Solution', 'accuracy_before', 'accuracy_after','weights_before',' weights_after', 'weights_percentage'])
df.to_csv(test_filename)

