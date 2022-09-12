import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import inspect
import importlib
import CompressionLibrary.CompressionTechniques as CompressionTechniques
from CompressionLibrary.CompressionTechniques import *
from CompressionLibrary.custom_callbacks import EarlyStoppingReward
from CompressionLibrary.utils import calculate_model_weights, extract_model_parts, create_model_from_parts
import logging

class ModelCompressionEnv():
    def __init__(self, compressors_list, create_model_func, compr_params,
                 train_ds, validation_ds, test_ds,
                 layer_name_list, input_shape, current_state_source='layer_input', next_state_source='layer_output', verbose=2, get_state_from='validation', tuning_epochs=20, num_feature_maps=128, tuning_batch_size=32, strategy=None, model_path='./data'):

        self._episode_ended = False
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.create_model_func = create_model_func
        self.train_ds = train_ds
        self.validation_ds = validation_ds
        self.test_ds = test_ds
        self.original_layer_name_list = layer_name_list
        self.layer_name_list = self.original_layer_name_list.copy()
        self.input_shape = input_shape
        self.current_state_source = current_state_source
        self.next_state_source = next_state_source
        self.compr_params = compr_params
        self.num_feature_maps = num_feature_maps
        self.tuning_batch_size = tuning_batch_size
        self.tuning_epochs = tuning_epochs
        self.get_state_from = get_state_from
        self.strategy = strategy
        self.callbacks = []

        self.model_path = model_path+'/temp_model'

        self.model = self.create_model_func()
        self.optimizer = tf.keras.optimizers.Adam(1e-5)
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
                               fine_tuning=False, input_shape=self.input_shape)

            if temp_comp.target_layer_type == 'conv':
                self.conv_compressors.append(compressor)
            elif temp_comp.target_layer_type == 'dense':
                self.dense_compressors.append(compressor)
            del temp_comp


        self.conv_compressors.remove('SparseConvolutionCompression')
        self.conv_compressors.remove('MLPCompression')
        self._layer_counter = 0

        max_filters = self.get_highest_num_filters()
        self.logger.debug('The highest number of filters of a Conv2D layer is {}.'.format(max_filters))
        self._state = self.get_state('current_state')
        self.conv_shape = self._state.shape

        self.conv_shape = tf.constant(self.conv_shape).numpy()
        
        self.conv_shape[-1] = max_filters
        

        self.dense_shape = self.get_output_feature_map('flatten').shape

        self.weights_before = int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))

        
        if self.strategy:
            self.logger.debug('Strategy found. Using strategy to evaluate.')
            layers, configs, weights = extract_model_parts(self.model)

            with self.strategy.scope():
                optimizer2 = tf.keras.optimizers.Adam(1e-5)
                loss2 = tf.keras.losses.SparseCategoricalCrossentropy()
                metric2 = tf.keras.metrics.SparseCategoricalAccuracy()
                self.model = create_model_from_parts(layers, configs, weights, optimizer2, loss2, metric2)
                self.logger.debug('Evaluating model using test set.')
                test_loss, self.test_acc_before = self.model.evaluate(self.test_ds, verbose=self.verbose)
                self.logger.info(f'Test accuracy is {self.test_acc_before} and loss {test_loss}')
                self.logger.debug('Evaluating model using validation set.')
                val_loss, self.val_acc_before = self.model.evaluate(self.validation_ds, verbose=self.verbose)
                self.logger.info(f'Test accuracy is {self.val_acc_before} and loss {val_loss}')
        else:

            self.logger.debug('Evaluating model using test set.')
            test_loss, self.test_acc_before = self.model.evaluate(self.test_ds, verbose=self.verbose)
            self.logger.info(f'Test accuracy is {self.test_acc_before} and loss {test_loss}')
            self.logger.debug('Evaluating model using validation set.')
            val_loss, self.val_acc_before = self.model.evaluate(self.validation_ds, verbose=self.verbose)
            self.logger.info(f'Test accuracy is {self.val_acc_before} and loss {val_loss}')

        self.logger.info('Finished environment initialization.')

    def get_highest_num_filters(self):
        filters = []
        previous_filters = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                config = layer.get_config()
                if layer.name in self.original_layer_name_list:
                    filters.append(previous_filters)
                    filters.append(config['filters'])
                else:
                    previous_filters = config['filters']
        return max(filters)

    def observation_space(self):
        return self.conv_shape, self.dense_shape

    def action_space(self):
        return len(self.conv_compressors) + 1, len(self.dense_compressors) + 1

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
                    self.logger.debug(f'Getting output feature map of layer {layer.name}')
                    output = layer(x)
                    break
                else:
                    x = layer(x)
        else:
            output = x

        generate_fmp = tf.keras.Model(inputs, output)
        generate_fmp.compile(optimizer=self.optimizer, loss=self.loss_object, metrics=self.train_metric)

        self.logger.debug('Finished creating model to extract feature map.')
        num_batches = self.num_feature_maps//self.tuning_batch_size

        if self.strategy:
            num_batches = 1

        self.logger.debug(f'For {self.num_feature_maps} samples, {num_batches} samples of {self.tuning_batch_size} are required.')

        if self.get_state_from == 'train':
            random_imgs = self.train_ds.take(num_batches)
        elif self.get_state_from == 'validation':
            random_imgs = self.validation_ds.take(num_batches)
        elif self.get_state_from == 'test':
            random_imgs = self.test_ds.take(num_batches)
        else:
            raise "Please choose from 'train', 'validation' and 'test'."

        self.logger.debug(f'Generating feature maps for layer {layer_name}')
        state = generate_fmp.predict(random_imgs)

        del generate_fmp

        return state

    def get_state(self, mode='current_state', offset=0):

        assert mode in ['current_state', 'next_state']
        if self._episode_ended:
            return None
        else:
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
            self.logger.debug(f'Getting  {mode} of layer {layer_name}.')
            self._state = self.get_output_feature_map(layer_name)
            self.logger.debug(f'State is {self.current_state_source}. Getting output of layer {layer_name} and has shape {self._state.shape}.')

            if hasattr(self, 'conv_shape'):
                if len(self._state.shape) == len(self.conv_shape):
                    b, h, w, c = self._state.shape
                    zeros = np.zeros(shape=(b,h,w,self.conv_shape[-1]))
                    zeros[:,:,:,:c] = self._state
                    self._state = zeros

                elif len(self._state.shape) == len(self.dense_shape):
                    b, units = self._state.shape
                    zeros = np.zeros(shape=(b,self.dense_shape[-1]))
                    zeros[:,:units] = self._state
                    self._state = zeros

            self.logger.debug(f'The state was reshaped into shape {self._state.shape}')

            return self._state

    def reset(self):
        self.logger.debug('---RESTARTING ENVIRONMENT---')

        self.model = self.create_model_func()
        self.layer_name_list = self.original_layer_name_list.copy()
        self.callbacks = []
        self._layer_counter = 0
        self._state = self.get_state('current_state')

        self._episode_ended = False
        self.weights_before = int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))

        self.chosen_actions = []

        return self._state

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        
        layer_name = self.layer_name_list[self._layer_counter]
        info = {'layer_name': layer_name}

        layer = self.model.get_layer(layer_name)
        if isinstance(layer, tf.keras.layers.Conv2D):
            compressors = self.conv_compressors
        else:
            compressors = self.dense_compressors

        self.logger.debug(f'Using action {action} on layer {layer_name}.')


        if action > len(compressors) + 1:
            action = 0
            info['action_overwritten'] = True
            self.chosen_actions.append('None')

        if action == 0:
            weight_diff = 0
            weights_before = self.weights_before
            weights_after = self.weights_before
            val_acc_after = self.val_acc_before
            test_acc_after = self.test_acc_before
            self.logger.debug(f'Layer {layer_name} was not compressed.')
            self.chosen_actions.append('None')

        else:
            action -= 1
            self.logger.debug(f'Compressing layer {layer_name} using {compressors[action]}')

            # Save the sequence of actions.
            self.chosen_actions.append(compressors[action])

            
            class_ = getattr(CompressionTechniques, compressors[action])

            compressor = class_(model=self.model, dataset=self.train_ds, optimizer=self.optimizer, loss=self.loss_object, metrics=self.train_metric, fine_tuning=False, input_shape=self.input_shape, tuning_verbose=self.verbose, callbacks=self.callbacks)


            weights_before = calculate_model_weights(self.model)

            compressor.weights_before = weights_before
            compressor.callbacks = self.callbacks

            if compressors[action] in self.compr_params.keys():
                # Replace target layer with current layer name.
                self.compr_params[compressors[action]]['layer_name'] = layer_name
                # Apply compressor to layer.
                compressor.compress_layer(**self.compr_params[compressors[action]])
            else:
                compressor.compress_layer(layer_name=layer_name)

            # Get compressed model
            self.model = compressor.get_model()

            self.callbacks = compressor.callbacks
            self.layer_name_list[self._layer_counter] = compressor.new_layer_name

            weights_after = calculate_model_weights(self.model)
            weight_diff = 1 - (weights_after / weights_before)
            val_acc_after = 0
            test_acc_after = 0

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
                Rcb = EarlyStoppingReward(weights_before=self.weights_before, verbose=1)
                self.callbacks.append(Rcb)

                if self.strategy:
                    self.logger.debug('Strategy found. Using strategy to fit model.')

                    # Extract core info of model to create another inside scope.
                    layers, configs, weights = extract_model_parts(self.model)

                    with self.strategy.scope():
                        optimizer2 = tf.keras.optimizers.Adam(1e-5)
                        loss2 = tf.keras.losses.SparseCategoricalCrossentropy()
                        metric2 = tf.keras.metrics.SparseCategoricalAccuracy()
                        self.model = create_model_from_parts(layers, configs, weights, optimizer2, loss2, metric2)
                        for layer in self.model.layers:
                            if layer.name in self.layer_name_list:
                                layer.trainable = True
                            else:
                                layer.trainable = False
                        self.model.summary()
                        self.model.fit(self.train_ds, epochs=self.tuning_epochs, callbacks=self.callbacks, validation_data=self.validation_ds)
                        test_loss, test_acc_after = self.model.evaluate(self.test_ds, verbose=self.verbose)
                        val_loss, val_acc_after = self.model.evaluate(self.validation_ds, verbose=self.verbose)

                    # Set all layers back to trainable.
                    for layer in self.model.layers:
                        layer.trainable = True

                else:

                    # Train only the modified layers.
                    for layer in self.model.layers:
                        if layer.name in self.layer_name_list:
                            layer.trainable = True
                        else:
                            layer.trainable = False
                    self.model.summary()
                    self.model.fit(self.train_ds, epochs=self.tuning_epochs, callbacks=self.callbacks, validation_data=self.validation_ds)

                    # Set all layers back to trainable.
                    for layer in self.model.layers:
                        layer.trainable = True

                    test_loss, test_acc_after = self.model.evaluate(self.test_ds, verbose=self.verbose)
                    val_loss, val_acc_after = self.model.evaluate(self.validation_ds, verbose=self.verbose)

        reward = weight_diff + test_acc_after
        info['test_acc_before'] = self.test_acc_before
        info['test_acc_after'] = test_acc_after
        info['weights_before'] = weights_before
        info['weights_after'] = weights_after
        info['val_acc_before'] = self.val_acc_before
        info['val_acc_after'] = val_acc_after
        info['actions'] = self.chosen_actions
        info['reward'] = reward
        

        return self._state, reward, self._episode_ended, info
