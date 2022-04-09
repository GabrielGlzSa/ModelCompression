from numpy.core.fromnumeric import compress
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import inspect
import importlib
from utils import load_dataset
import CompressionTechniques
from CompressionTechniques import *
import logging


class LayerEnv():
    def __init__(self, compressors_list, model_path, compr_params,
                 dataset, validation,
                 layer_name_list, input_shape, features_type='weight_shape'):

        self.model_path = model_path
        self.dataset = dataset
        self.validation = validation
        self.input_shape = input_shape
        self.layer_name_list = layer_name_list
        self.compr_params = compr_params
        self.model = tf.keras.models.load_model(model_path, compile=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        compressors = [name for name, cls in
                       inspect.getmembers(importlib.import_module("CompressionTechniques"), inspect.isclass) if
                       issubclass(cls, ModelCompression)]

        self.features_type = features_type
        self.conv_compressors = []
        self.dense_compressors = []
        for compressor in compressors:
            if compressor not in compressors_list:
                continue
            class_ = getattr(CompressionTechniques, compressor)
            temp_comp = class_(model=self.model, dataset=self.dataset, optimizer=self.optimizer, loss=self.loss_object,
                               metrics=self.train_metric,
                               fine_tune=True, input_shape=self.input_shape)
            if temp_comp.target_layer_type == 'conv':
                self.conv_compressors.append(compressor)
            elif temp_comp.target_layer_type == 'dense':
                self.dense_compressors.append(compressor)

        self.model = tf.keras.models.load_model(model_path, compile=True)

        if self.features_type == 'weights_shape':
            self._state = self.model.get_layer(self.layer_name_list[0]).get_weights()[0].shape
        elif self.features_type == 'input_shape':
            self._state = self.model.get_layer(self.layer_name_list[0]).input_shape[1:]
            print(self._state)

        self._layer_counter = 0

        self._episode_ended = False
        self.weights_before = int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))
        loss, self.acc_before = self.model.evaluate(self.validation, verbose=0)

    def observation_space(self):
        return len(self._state)

    def action_space(self, layer_type='conv'):
        if layer_type == 'conv':
            return len(self.conv_compressors) + 1
        else:
            return len(self.dense_compressors) + 1

    # def action_space_fc(self):
    #     layer = self.model.get_layer(self.layer_name_list[self._layer_counter])
    #     if isinstance(layer, tf.keras.layers.Conv2D):
    #         return len(self.conv_compressors) + 1
    #     elif isinstance(layer, tf.keras.layers.Dense):
    #         return len(self.dense_compressors) + 1
    #     else:
    #         raise "Unsupported type of layer. Only Conv2D and Dense layers are supported."

    def next_layer(self):
        return self.layer_name_list[self._layer_counter]

    def reset(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=True)
        if self.features_type == 'weights_shape':
            self._state = self.model.get_layer(self.layer_name_list[0]).get_weights()[0].shape
        elif self.features_type == 'input_shape':
            self._state = self.model.get_layer(self.layer_name_list[0]).input_shape[1:]

        self._layer_counter = 0
        self._episode_ended = False
        self.weights_before = int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))

        return self._state

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        info = {}
        layer_name = self.layer_name_list[self._layer_counter]
        layer = self.model.get_layer(layer_name)
        if isinstance(layer, tf.keras.layers.Conv2D):
            compressors = self.conv_compressors
        else:
            compressors = self.dense_compressors

        if action > len(compressors) + 1:
            action = 0
            info['action_overwritten'] = True

        if action == 0:
            weight_diff = 0
            weights_before = self.weights_before
            weights_after = self.weights_before
            val_acc_after = self.acc_before


        else:
            # print('action_before:', action )
            action -= 1
            # print(compressors)
            # print(compressors[action])

            class_ = getattr(CompressionTechniques, compressors[action])
            compressor = class_(model=self.model, dataset=self.dataset, optimizer=self.optimizer, loss=self.loss_object,
                                metrics=self.train_metric,
                                fine_tune=True, input_shape=self.input_shape)

            compressor.weights_before = self.weights_before

            if compressors[action] in self.compr_params.keys():
                self.compr_params[compressors[action]]['layer_name'] = layer_name
                compressor.compress_layer(**self.compr_params[compressors[action]])
            else:
                compressor.compress_layer(layer_name=layer_name)

            self.model = compressor.get_model()

            weights_before, weights_after = compressor.get_weights_diff()
            weight_diff = 1 - (weights_after / weights_before)

            loss, val_acc_after = self.model.evaluate(self.validation, verbose=0)

        info['acc_before'] = self.acc_before
        info['acc_after'] = val_acc_after
        info['weights_before'] = weights_before
        info['weights_after'] = weights_after

        self.weights_before = weights_after
        self.acc_before = val_acc_after
        reward = weight_diff + val_acc_after

        if compressors[action] == 'ReplaceDenseWithGlobalAvgPool':
            self._episode_ended = True
            self._state = self.model.layers[-1].get_weights()[0].shape
            if self.features_type == 'weights_shape':
                self._state = self.model.layers[-1].get_weights()[0].shape
            elif self.features_type == 'input_shape':
                self._state = self.model.layers[-1].input_shape[1:]
                self._state = tuple(list(self._state) + [0, 0])
        else:
            self._layer_counter += 1
            if self._layer_counter == len(self.layer_name_list):
                self._episode_ended = True
            else:
                self._state = self.model.get_layer(self.layer_name_list[self._layer_counter]).get_weights()[0].shape
                if self.features_type == 'weights_shape':
                    self._state = self.model.get_layer(self.layer_name_list[self._layer_counter]).get_weights()[0].shape
                elif self.features_type == 'input_shape':
                    self._state = self.model.get_layer(self.layer_name_list[self._layer_counter]).input_shape[1:]
                    self._state = tuple(list(self._state) + [0, 0])

        if self.features_type == 'weights_shape' and len(self._state) == 2:
            self._state = tuple(list(self._state) + [0, 0])
        return self._state, reward, self._episode_ended, info


class AdaDeepEnv():
    def __init__(self, compressors_list, model_path, compr_params,
                 dataset, validation,
                 layer_name_list, input_shape, current_state_source='layer_input', next_state_source='layer_output'):

        self._episode_ended = False
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.dataset = dataset
        self.validation = validation
        self.original_layer_name_list = layer_name_list
        self.layer_name_list = self.original_layer_name_list.copy()
        self.input_shape = input_shape
        self.current_state_source = current_state_source
        self.next_state_source = next_state_source
        self.compr_params = compr_params
        self.model = tf.keras.models.load_model(model_path, compile=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        compressors = [name for name, cls in
                       inspect.getmembers(importlib.import_module("CompressionTechniques"), inspect.isclass) if
                       issubclass(cls, ModelCompression)]

        self.conv_compressors = []
        self.dense_compressors = []
        for compressor in compressors:
            if compressor not in compressors_list:
                continue
            class_ = getattr(CompressionTechniques, compressor)
            temp_comp = class_(model=self.model, dataset=self.dataset, optimizer=self.optimizer, loss=self.loss_object,
                               metrics=self.train_metric,
                               fine_tune=True, input_shape=self.input_shape)
            if temp_comp.target_layer_type == 'conv':
                self.conv_compressors.append(compressor)
            elif temp_comp.target_layer_type == 'dense':
                self.dense_compressors.append(compressor)

        self.model = tf.keras.models.load_model(model_path, compile=True)

        self._layer_counter = 0

        self._state = self.get_current_state()

        self.weights_before = int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))
        loss, self.acc_before = self.model.evaluate(self.validation, verbose=0)

    def observation_space(self):
        return len(self._state)

    def action_space(self, layer_type='conv'):
        if layer_type == 'conv':
            return len(self.conv_compressors) + 1
        else:
            return len(self.dense_compressors) + 1

    def next_layer(self):
        return self.layer_name_list[self._layer_counter]

    def get_state_input(self):
        names = [layer.name for layer in self.model.layers]
        self.logger.debug('Layers are: {}'.format(names))
        layer_idx = names.index(self.layer_name_list[self._layer_counter]) - 1
        layer_name = names[layer_idx]
        self.logger.debug('Current state is the output of {} layer.'.format(layer_name))

        generate_input = tf.keras.Model(self.model.input, self.model.get_layer(layer_name).output)
        state = generate_input.predict(self.validation)
        return state

    def get_state_output(self):
        names = [layer.name for layer in self.model.layers]
        self.logger.debug('Layers are: {}'.format(names))
        layer_idx = names.index(self.layer_name_list[self._layer_counter])
        layer_name = names[layer_idx]
        self.logger.debug('Next state is the output of {} layer.'.format(layer_name))
        output = self.model.layers[layer_idx].output
        generate_input = tf.keras.Model(self.model.input, self.model.get_layer(layer_name).output)

        state = generate_input.predict(self.validation)
        return state

    def get_current_state(self):
        if self._episode_ended:
            return None
        else:
            self.model.summary()
            if self.current_state_source == 'layer_input':
                self._state = self.get_state_input()
            elif self.current_state_source == 'layer_output':
                self._state = self.get_state_output()

            self.logger.debug('State has shape {}.'.format(self._state.shape))

            return self._state

    def reset(self):
        self.model.summary()
        self.logger.debug('---RESTARTING ENVIRONMENT---')

        self.model = tf.keras.models.load_model(self.model_path, compile=True)
        self.layer_name_list = self.original_layer_name_list.copy()

        self._layer_counter = 0
        if self.current_state_source == 'layer_input':
            self._state = self.get_state_input()
        elif self.current_state_source == 'layer_output':
            self._state = self.get_state_output()

        self._episode_ended = False
        self.weights_before = int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))

        return self._state

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        info = {}
        layer_name = self.layer_name_list[self._layer_counter]
        layer = self.model.get_layer(layer_name)
        if isinstance(layer, tf.keras.layers.Conv2D):
            compressors = self.conv_compressors
        else:
            compressors = self.dense_compressors

        self.logger.debug('Using action {} on layer {}.'.format(action, layer_name))
        if action > len(compressors) + 1:
            action = 0
            info['action_overwritten'] = True

        if action == 0:
            weight_diff = 0
            weights_before = self.weights_before
            weights_after = self.weights_before
            val_acc_after = self.acc_before
            self.logger.debug('Layer {} was not compressed.'.format(layer_name))


        else:
            action -= 1
            self.logger.debug('Compressing layer {} using {}'.format(layer_name, compressors[action]))

            class_ = getattr(CompressionTechniques, compressors[action])
            compressor = class_(model=self.model, dataset=self.dataset, optimizer=self.optimizer, loss=self.loss_object,
                                metrics=self.train_metric,
                                fine_tune=True, input_shape=self.input_shape)

            compressor.weights_before = self.weights_before

            if compressors[action] in self.compr_params.keys():
                self.compr_params[compressors[action]]['layer_name'] = layer_name
                compressor.compress_layer(**self.compr_params[compressors[action]])
            else:
                compressor.compress_layer(layer_name=layer_name)

            self.model = compressor.get_model()

            self.layer_name_list[self._layer_counter] = compressor.new_layer_name

            weights_before, weights_after = compressor.get_weights_diff()
            weight_diff = 1 - (weights_after / weights_before)

            loss, val_acc_after = self.model.evaluate(self.validation, verbose=0)

        info['acc_before'] = self.acc_before
        info['acc_after'] = val_acc_after
        info['weights_before'] = weights_before
        info['weights_after'] = weights_after

        self.weights_before = weights_after
        self.acc_before = val_acc_after
        reward = weight_diff + val_acc_after

        if compressors[action] == 'ReplaceDenseWithGlobalAvgPool':
            self._episode_ended = True

            generate_input = tf.keras.Model(self.model.input, self.model.layers[-2].output)

            self._state = generate_input.predict(self.validation)
        else:
            if self.next_state_source == 'layer_output':
                self._state = self.get_state_output()
            elif self.next_state_source == 'next_layer_input':
                self._state = self.get_state_input(i=1)

            self._layer_counter += 1
            if self._layer_counter == len(self.layer_name_list):
                self._episode_ended = True

        return self._state, reward, self._episode_ended, info
