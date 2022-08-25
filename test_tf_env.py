from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tabnanny import verbose

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import inspect
import importlib
import CompressionLibrary.CompressionTechniques as CompressionTechniques
from CompressionLibrary.CompressionTechniques import *
from CompressionLibrary.utils import calculate_model_weights, PCA
from CompressionLibrary.custom_callbacks import EarlyStoppingReward
import logging
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


class ModelCompressionEnvTFA(py_environment.PyEnvironment):
    def __init__(self,compressors_list, compr_params,
                 train_ds, validation_ds, test_ds, 
                 layer_name_list, input_shape, fine_tuning_epochs=6, verbose=False, get_state_from='validation'):
        self._episode_ended = False
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.train_ds = train_ds
        self.validation_ds = validation_ds
        self.test_ds = test_ds
        self.original_layer_name_list = layer_name_list
        self.layer_name_list = self.original_layer_name_list.copy()
        self.input_shape = input_shape
        self.fine_tuning_epochs=fine_tuning_epochs
        self.compr_params = compr_params
        self.get_state_from = get_state_from

        self.model = tf.keras.applications.vgg16.VGG16(
                        include_top=True,
                        weights='imagenet',
                        input_shape=(224,224,3),
                        classes=1000,
                        classifier_activation='softmax'
                    )
                    
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metrics = tf.keras.metrics.SparseCategoricalAccuracy()

        self.model.compile(optimizer=self.optimizer, loss=self.loss_object, metrics=self.metrics)

        self.chosen_actions = []

        compressors = [name for name, cls in
                       inspect.getmembers(importlib.import_module("CompressionLibrary.CompressionTechniques"), inspect.isclass) if
                       issubclass(cls, ModelCompression)]

        self.conv_compressors = []
        self.dense_compressors = []
        for compressor in compressors:
            if compressor not in compressors_list:
                continue
            class_ = getattr(CompressionTechniques, compressor)
            temp_comp = class_(model=self.model, dataset=self.train_ds, optimizer=self.optimizer, loss=self.loss_object, metrics=self.metrics,
                               fine_tuning=False, input_shape=self.input_shape)
            if temp_comp.target_layer_type == 'conv':
                self.conv_compressors.append(compressor)
            elif temp_comp.target_layer_type == 'dense':
                self.dense_compressors.append(compressor)

        self._layer_counter = 0

        self.weights_shape = self.get_shape_most_weights()
        self.logger.debug(f'The highest number of weights is in shape {self.weights_shape}.')

        self.max_shape = None

        max_weights = 0
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                kernel = layer.get_weights()[0]
                _, _, channels, _ = kernel.shape
                n_weights = tf.size(kernel)
                if n_weights > max_weights:
                    max_weights = n_weights
                    self.max_shape = kernel.shape
            if isinstance(layer, tf.keras.layers.Flatten):
                break
        
        h, w, c, filters = self.max_shape
        self.observation_shape = (filters, h*w, c)

        self._state = self.get_weight_state('current_state')

        self.weights_before = int(
            np.sum([K.count_params(w) for w in self.model.trainable_weights]))

        self.original_num_weights = self.weights_before

        loss, self.acc_before = self.model.evaluate(
            self.validation_ds, verbose=0)

        test_loss, self.acc_before = self.model.evaluate(self.test_ds, verbose=0)
        val_loss, self.val_acc_before = self.model.evaluate(self.validation_ds, verbose=0)

        max_actions = max(len(self.conv_compressors), len(self.dense_compressors))

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=max_actions, name='action')

        self._observation_spec = array_spec.ArraySpec(
            shape=self.observation_shape, dtype=np.float32, name='observation_spec')


    def get_shape_most_weights(self):
        weights = [np.sum([K.count_params(w) for w in layer.trainable_weights]) for layer in self.model.layers]
        idx = np.argmax(weights)
        layer = self.model.layers[idx]
        weights, _ = layer.get_weights()
        return weights.shape

    def action_spec(self):
        return self._action_spec
       
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        # Restart model
        self.model = tf.keras.applications.vgg16.VGG16(
                        include_top=True,
                        weights='imagenet',
                        input_shape=(224,224,3),
                        classes=1000,
                        classifier_activation='softmax'
                    )

        self.model.compile(optimizer=self.optimizer, loss=self.loss_object, metrics=self.metrics)

        # Restart layer list.
        self.layer_name_list = self.original_layer_name_list.copy()

        # Next layer to process is the first layer of the list.
        self._layer_counter = 0
        # Get state for first layer of the list.
        self._state = self.get_weight_state('current_state')
        
        self._episode_ended = False

        # Count number of weights
        self.weights_before = int(
            np.sum([K.count_params(w) for w in self.model.trainable_weights]))

        self.chosen_actions = []

        return ts.restart(np.array([self._state], dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        info = {}
        layer_name = self.layer_name_list[self._layer_counter]
        layer = self.model.get_layer(layer_name)
        if isinstance(layer, tf.keras.layers.Conv2D):
            compressors = self.conv_compressors
            info['was_conv'] = True
        else:
            compressors = self.dense_compressors
            info['was_conv'] = False

        self.logger.debug(
            'Using action {} on layer {}.'.format(action, layer_name))

        if action > len(compressors) + 1:
            action = 0
            info['action_overwritten'] = True
            self.chosen_actions.append('None')

        if action == 0:
            weight_diff = 0
            weights_before = self.weights_before
            weights_after = self.weights_before
            val_acc_after = self.val_acc_before
            test_acc_after = self.acc_before
            self.logger.debug(
                'Layer {} was not compressed.'.format(layer_name))
            self.chosen_actions.append('None')

        else:
            action -= 1
            self.logger.debug('Compressing layer {} using {}'.format(
                layer_name, compressors[action]))

            self.chosen_actions.append(compressors[action])

            class_ = getattr(CompressionTechniques, compressors[action])
            compressor = class_(model=self.model, dataset=self.train_ds, optimizer=self.optimizer,
                                loss=self.loss_object, metrics=self.metrics, fine_tune=True, input_shape=self.input_shape, verbose=0)

            compressor.weights_before = self.weights_before

            if compressors[action] in self.compr_params.keys():
                self.compr_params[compressors[action]
                                  ]['layer_name'] = layer_name
                compressor.compress_layer(
                    **self.compr_params[compressors[action]])
            else:
                compressor.compress_layer(layer_name=layer_name)

            self.model = compressor.get_model()

            self.layer_name_list[self._layer_counter] = compressor.new_layer_name

            weights_before, weights_after = compressor.get_weights_diff()
            weight_diff = 1 - (weights_after / weights_before)

            test_loss, test_acc_after = self.model.evaluate(self.test_ds, verbose=0)
            val_loss, val_acc_after = self.model.evaluate(self.validation_ds, verbose=0)

        info['acc_before'] = self.acc_before
        info['acc_after'] = val_acc_after
        info['weights_before'] = weights_before
        info['weights_after'] = weights_after
        info['val_acc_before'] = self.val_acc_before
        info['val_acc_after'] = val_acc_after
        info['actions'] = self.chosen_actions

        self.weights_before = weights_after
        self.acc_before = test_acc_after
        reward = weight_diff + test_acc_after
        self.acc_before_val = val_acc_after

        if compressors[action] == 'ReplaceDenseWithGlobalAvgPool':
            self._episode_ended = True
            current_idx = self._layer_counter
            offset = len(self.model.layers[current_idx:-2])
            self._state = self.get_weight_state(mode='next_state', offset=offset)
        else:
            self._state = self.get_weight_state('next_state')
            self._layer_counter += 1
            if self._layer_counter == len(self.layer_name_list):
                self._episode_ended = True
                
                cb = compressor.callbacks
                if cb is None:
                    cb = []
                Rcb = EarlyStoppingReward(acc_before=self.acc_before_val, weights_before=self.original_num_weights)
                cb.append(Rcb)
                self.model.fit(train_ds, epochs=self.fine_tuning_epochs, validation_data=valid_ds, callbacks=cb)
                loss, valid_acc = self.model.evaluate(valid_ds, verbose=0)
                weights_after = calculate_model_weights(model)
                valid_reward = 1 - (weights_after/weights_before) + valid_acc

                print(f'Validation reward is {valid_reward}. The model has {weights_after} weights and {valid_acc} accuracy.')
                loss, test_acc = self.model.evaluate(test_ds)
                test_reward = 1 - (weights_after/weights_before) + test_acc
                print(f'Validation reward is {test_reward}. The model has {weights_after} weights and {test_acc} accuracy.')
                return ts.termination(self._state, reward)

        return ts.transition(self._state, reward=0.0, discount=1.0)


    def next_layer(self):
        return self.layer_name_list[self._layer_counter]

    def get_weight_state(self, mode='current_state', offset=0):

        assert mode in ['current_state', 'next_state']
        if self._episode_ended:
            return None
        else:
            names = [layer.name for layer in self.model.layers]
            self.logger.debug('Layers are: {}'.format(names))
            if mode == 'current_state':
                 layer_idx = names.index(self.layer_name_list[self._layer_counter])
            if mode == 'next_state':
                layer_idx = names.index(self.layer_name_list[self._layer_counter+1])

            layer_name = names[layer_idx+offset]
            self.logger.debug(f'Getting  {mode} of layer {layer_name}.')
            self._state = self.model.get_layer(layer_name).get_weights()[0]
            self.logger.debug(f'Getting weights of layer {layer_name} with shape {self._state.shape}')

            layer = self.model.get_layer(layer_name)

            if isinstance(layer, tf.keras.layers.Conv2D):
                self._state = tf.transpose(self._state, perm=[3,0,1,2])

                f, h, w, c = self._state.shape
                self._state = tf.reshape(self._state, shape=[f, -1, c])
                h, w, c = self._state.shape

                indexes = [[x,y,z] for x in range(h) for y in range(w) for z in range(c)]
                indexes = tf.constant(indexes)
                self._state = tf.scatter_nd(indexes, tf.reshape(self._state, shape=-1), self.observation_shape)
            else:
                h, w, c = tf.shape(self._state)
                indexes = [[x,y,z] for x in range(h) for y in range(w) for z in range(c)]
                indexes = tf.constant(indexes)

                self._state = tf.scatter_nd(indexes, tf.reshape(self._state, shape=-1), self.observation_shape)

           
     
            self.logger.debug(f'The state was reshaped into shape {self._state.shape}')


            return self._state


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

input_shape = (224,224,3)

optimizer = tf.keras.optimizers.Adam(1e-5)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

batch_size = 64

model = tf.keras.applications.vgg16.VGG16(
                        include_top=True,
                        weights='imagenet',
                        input_shape=(224,224,3),
                        classes=num_classes,
                        classifier_activation='softmax'
                    )


start = time.time()

train_ds = train_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=2000, reshuffle_each_iteration=True).batch(batch_size).take(10).prefetch(tf.data.AUTOTUNE)
valid_ds = validation_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).take(10).prefetch(tf.data.AUTOTUNE)
test_ds = test_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)



model.compile(optimizer=optimizer, loss=loss_object,
                metrics=train_metric)

model.summary()
layer_name_list = [ 'block2_conv1',  'block3_conv2','fc1', 'fc2']
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)


w_comprs = ['InsertDenseSVD', 'InsertDenseSparse',
            'DeepCompression'] 
c_comprs = ['InsertSVDConv', 'SparseConvolutionCompression',
            'DepthwiseSeparableConvolution', 'SparseConnectionsCompression']
l_comprs = ['FireLayerCompression', 'MLPCompression',
            'ReplaceDenseWithGlobalAvgPool']
compressors_list = w_comprs + c_comprs + l_comprs

parameters = {}
parameters['DeepCompression'] = {
    'layer_name': None, 'threshold': 0.001}
parameters['ReplaceDenseWithGlobalAvgPool'] = {'layer_name': None}
parameters['InsertDenseSVD'] = {'layer_name': None}
parameters['InsertDenseSparse'] = {'layer_name': None}
parameters['InsertSVDConv'] = {'layer_name': None}
parameters['DepthwiseSeparableConvolution'] = {'layer_name': None}
parameters['FireLayerCompression'] = {'layer_name': None}
parameters['MLPCompression'] = {'layer_name': None}
parameters['SparseConvolutionCompression'] = {'layer_name': None}
parameters['SparseConnectionsCompression'] = {'layer_name': None, 'new_layer_iterations': 20,
                                                'target_perc': 0.75, 'conn_perc_per_epoch': 0.15}


# environment = ModelCompressionEnvTFA(compressors_list, parameters,
#                                     train_ds, valid_ds, test_ds,
#                                     layer_name_list, input_shape, verbose=False)

# print('action_spec:', environment.action_spec())
# print('time_step_spec.observation:', environment.time_step_spec().observation)
# print('time_step_spec.step_type:', environment.time_step_spec().step_type)
# print('time_step_spec.discount:', environment.time_step_spec().discount)
# print('time_step_spec.reward:', environment.time_step_spec().reward)


# utils.validate_py_environment(environment, episodes=2)

# action = np.array(1, dtype=np.int32)

# time_step = environment.reset()
# print(time_step)
# cumulative_reward = time_step.reward

# for _ in range(5):
#   time_step = environment.step(action)
#   print(time_step)
#   cumulative_reward += time_step.reward

# print('Final Reward = ', cumulative_reward)

X = tf.transpose(model.get_layer('fc1').get_weights()[0])
print(X.shape)
def mse(predict, actual):
    """Helper function for computing the mean squared error (MSE)"""
    return np.square(predict - actual).sum(axis=1).mean()

loss = []
reconstructions = []
# iterate over different number of principal components, and compute the MSE
for num_component in range(1, 1000, 100):
    reconst, _, _, _ = PCA(X, num_component, high_dim=True)
    error = mse(reconst, X)
    reconstructions.append(reconst)
    print('n = {:d}, reconstruction_error = {:f}'.format(num_component, error))
    
    loss.append((num_component, error))

reconstructions = np.asarray(reconstructions)
reconstructions = reconstructions
loss = np.asarray(loss)


# create a table showing the number of principal components and MSE
df = pd.DataFrame(loss, columns=['no. of components', 'mse'])
print(df.to_string())
