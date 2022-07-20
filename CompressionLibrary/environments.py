from tabnanny import verbose
from numpy.core.fromnumeric import compress
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import inspect
import importlib
import CompressionLibrary.CompressionTechniques as CompressionTechniques
from CompressionLibrary.CompressionTechniques import *
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
            temp_comp = class_(model=self.model, dataset=self.dataset, optimizer=self.optimizer, loss=self.loss_object, metrics=self.train_metric,
                               fine_tune=True, input_shape=self.input_shape)
            if temp_comp.target_layer_type == 'conv':
                self.conv_compressors.append(compressor)
            elif temp_comp.target_layer_type == 'dense':
                self.dense_compressors.append(compressor)

        self.model = tf.keras.models.load_model(model_path, compile=True)

        if self.features_type == 'weights_shape':
            self._state = self.model.get_layer(
                self.layer_name_list[0]).get_weights()[0].shape
        elif self.features_type == 'input_shape':
            self._state = self.model.get_layer(
                self.layer_name_list[0]).input_shape[1:]
            print(self._state)

        self._layer_counter = 0

        self._episode_ended = False
        self.weights_before = int(
            np.sum([K.count_params(w) for w in self.model.trainable_weights]))
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

    def reset(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=True)
        if self.features_type == 'weights_shape':
            self._state = self.model.get_layer(
                self.layer_name_list[0]).get_weights()[0].shape
        elif self.features_type == 'input_shape':
            self._state = self.model.get_layer(
                self.layer_name_list[0]).input_shape[1:]

        self._layer_counter = 0
        self._episode_ended = False
        self.weights_before = int(
            np.sum([K.count_params(w) for w in self.model.trainable_weights]))

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
            compressor = class_(model=self.model, dataset=self.dataset, optimizer=self.optimizer, loss=self.loss_object, metrics=self.train_metric,
                                fine_tune=True, input_shape=self.input_shape)

            compressor.weights_before = self.weights_before

            if compressors[action] in self.compr_params.keys():
                self.compr_params[compressors[action]
                                  ]['layer_name'] = layer_name
                compressor.compress_layer(
                    **self.compr_params[compressors[action]])
            else:
                compressor.compress_layer(layer_name=layer_name)

            self.model = compressor.get_model()

            weights_before, weights_after = compressor.get_weights_diff()
            weight_diff = 1 - (weights_after / weights_before)

            loss, val_acc_after = self.model.evaluate(
                self.validation, verbose=0)

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
                self._state = tuple(list(self._state)+[0, 0])
        else:
            self._layer_counter += 1
            if self._layer_counter == len(self.layer_name_list):
                self._episode_ended = True
            else:
                self._state = self.model.get_layer(
                    self.layer_name_list[self._layer_counter]).get_weights()[0].shape
                if self.features_type == 'weights_shape':
                    self._state = self.model.get_layer(
                        self.layer_name_list[self._layer_counter]).get_weights()[0].shape
                elif self.features_type == 'input_shape':
                    self._state = self.model.get_layer(
                        self.layer_name_list[self._layer_counter]).input_shape[1:]
                    self._state = tuple(list(self._state)+[0, 0])

        if self.features_type == 'weights_shape' and len(self._state) == 2:
            self._state = tuple(list(self._state)+[0, 0])
        return self._state, reward, self._episode_ended, info


class ModelCompressionEnv():
    def __init__(self, compressors_list, model_path, compr_params,
                 train_ds, validation_ds, test_ds,
                 layer_name_list, input_shape, current_state_source='layer_input', next_state_source='layer_output', verbose=False):

        self._episode_ended = False
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.train_ds = train_ds
        self.validation_ds = validation_ds
        self.test_ds = test_ds
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
            temp_comp = class_(model=self.model, dataset=self.train_ds, optimizer=self.optimizer, loss=self.loss_object, metrics=self.train_metric,
                               fine_tune=True, input_shape=self.input_shape)
            if temp_comp.target_layer_type == 'conv':
                self.conv_compressors.append(compressor)
            elif temp_comp.target_layer_type == 'dense':
                self.dense_compressors.append(compressor)

        self.model = tf.keras.models.load_model(model_path, compile=True)

        self._layer_counter = 0

        self._state = self.get_state('current_state')
        self.conv_shape = self._state.shape

        self.dense_shape = self.get_output_feature_map('flatten').shape

        self.weights_before = int(
            np.sum([K.count_params(w) for w in self.model.trainable_weights]))
        loss, self.acc_before = self.model.evaluate(
            self.validation_ds, verbose=0)

        test_loss, self.acc_before = self.model.evaluate(self.test_ds, verbose=0)
        val_loss, self.val_acc_before = self.model.evaluate(self.validation_ds, verbose=0)

    def observation_space(self):
        return len(self._state)

    def action_space(self, layer_type='conv'):
        if layer_type == 'conv':
            return len(self.conv_compressors) + 1
        else:
            return len(self.dense_compressors) + 1

    def next_layer(self):
        return self.layer_name_list[self._layer_counter]

    def get_output_feature_map(self, layer_name):

        inputs = tf.keras.layers.Input(shape=self.input_shape)

        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        names = [layer.name for layer in self.model.layers[start:]]
        if layer_name in names:
            for layer in self.model.layers[start:]:
                if layer.name == layer_name:
                    self.logger.debug(
                        'Getting output feature map of layer {}'.format(layer.name))
                    output = layer(x)
                    break
                else:
                    x = layer(x)
        else:
            output = x

        generate_fmp = tf.keras.Model(inputs, output)

        state = generate_fmp.predict(self.validation_ds)

        del generate_fmp
        return state

    def get_state(self, mode='current_state', offset=0):

        assert mode in ['current_state', 'next_state']
        if self._episode_ended:
            return None
        else:
            # self.model.summary()
            
            names = [layer.name for layer in self.model.layers]
            self.logger.debug('Layers are: {}'.format(names))
            if mode == 'current_state':
                if self.current_state_source == 'layer_input':
                    layer_idx = names.index(
                        self.layer_name_list[self._layer_counter]) - 1
                elif self.current_state_source == 'layer_output':
                    layer_idx = names.index(
                        self.layer_name_list[self._layer_counter])
                elif self.current_state_source == 'layer_weights':
                    pass
            if mode == 'next_state':
                if self.next_state_source == 'layer_input':
                    layer_idx = names.index(
                        self.layer_name_list[self._layer_counter]) - 1
                elif self.next_state_source == 'layer_output':
                    layer_idx = names.index(
                        self.layer_name_list[self._layer_counter])
                elif self.next_state_source == 'layer_weights':
                    pass

            layer_name = names[layer_idx+offset]
            self.logger.debug('Getting  {} of layer {}.'.format(mode, layer_name))
            self._state = self.get_output_feature_map(layer_name)
            self.logger.debug('State is {}. Getting output of layer {} and has shape {}'.format(
                self.current_state_source, layer_name, self._state.shape))

            if hasattr(self, 'conv_shape'):
                if len(self._state.shape) == len(self.conv_shape):
                    b, h, w, c = self._state.shape
                    zeros = np.zeros(shape=self.conv_shape)
                    zeros[:, :h, :w, :c] = self._state
                    self._state = zeros
                    
                elif len(self._state.shape) == len(self.dense_shape):
                    batch, units = self._state.shape
                    zeros = np.zeros(shape=self.dense_shape)
                    zeros[: , :units] = self._state
                    self._state = zeros
     
            self.logger.debug('State is {}. Getting output of layer {} and has shape {}'.format(
                    self.current_state_source, layer_name, self._state.shape))

            return self._state

    def reset(self):
        # self.model.summary()
        self.logger.debug('---RESTARTING ENVIRONMENT---')

        self.model = tf.keras.models.load_model(self.model_path, compile=True)
        self.layer_name_list = self.original_layer_name_list.copy()

        self._layer_counter = 0
        self._state = self.get_state('current_state')

        self._episode_ended = False
        self.weights_before = int(
            np.sum([K.count_params(w) for w in self.model.trainable_weights]))

        self.chosen_actions = []

        return self._state

    def step(self, action):
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

        else:
            action -= 1
            self.logger.debug('Compressing layer {} using {}'.format(
                layer_name, compressors[action]))

            self.chosen_actions.append(compressors[action])

            class_ = getattr(CompressionTechniques, compressors[action])
            compressor = class_(model=self.model, dataset=self.train_ds, optimizer=self.optimizer,
                                loss=self.loss_object, metrics=self.train_metric, fine_tune=True, input_shape=self.input_shape, verbose=0)

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
            self._state = self.get_state(mode='next_state', offset=offset)
        else:
            self._state = self.get_state('next_state')
            self._layer_counter += 1
            if self._layer_counter == len(self.layer_name_list):
                self._episode_ended = True

        return self._state, reward, self._episode_ended, info
