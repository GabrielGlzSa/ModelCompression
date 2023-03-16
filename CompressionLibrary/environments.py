import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import inspect
import importlib
import CompressionLibrary.CompressionTechniques as CompressionTechniques
from CompressionLibrary.CompressionTechniques import *
from CompressionLibrary.custom_callbacks import RestoreBestWeights
from CompressionLibrary.utils import calculate_model_weights, extract_model_parts, create_model_from_parts
import logging
import copy

class ModelCompressionEnv():
    def __init__(self, reward_func, compressors_list, create_model_func, compr_params,
                 train_ds, validation_ds, test_ds,
                 layer_name_list, input_shape, current_state_source='layer_input', next_state_source='layer_output', verbose=0, get_state_from='validation', tuning_mode='layer',tuning_epochs=5, num_feature_maps=128, tuning_batch_size=32, strategy=None, model_path='./data'):

        self.reward_func = reward_func
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
        self.tuning_mode = tuning_mode
        self.callbacks = []
        self.current_batch = None


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

        self.logger.info(f'There are {len(self.conv_compressors)} conv and {len(self.dense_compressors)} dense compressors.')

        self._layer_counter = 0


        if self.current_state_source == 'layer_weights':
            self.conv_shape = (None, None, 1)
            self.dense_shape = (None, None, 1)
        else:
            max_filters = self.get_highest_num_filters()
            self.logger.debug('The highest number of filters of a Conv2D layer is {}.'.format(max_filters))
            self._state = self.get_state('current_state')
            self.conv_shape = self._state.shape

            
            self.conv_shape = tf.constant(self.conv_shape).numpy()
            
            self.conv_shape[-1] = max_filters
            

            self.dense_shape = self.get_output_feature_map('flatten').shape

        self.weights_before = int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))
        self.weights_previous_it = self.weights_before
        
        if self.strategy:
            self.logger.debug('Strategy found. Using strategy to evaluate.')
            layers, configs, weights = extract_model_parts(self.model)

            with self.strategy.scope():
                optimizer2 = tf.keras.optimizers.Adam(1e-5)
                loss2 = tf.keras.losses.SparseCategoricalCrossentropy()
                metric2 = tf.keras.metrics.SparseCategoricalAccuracy()
                self.model = create_model_from_parts(layers, configs, weights, optimizer2, loss2, metric2, input_shape=self.input_shape)
                self.logger.debug('Evaluating model using test set.')
                test_loss, self.test_acc_before = self.model.evaluate(self.test_ds, verbose=self.verbose)
                self.logger.info(f'Test accuracy is {self.test_acc_before} and loss {test_loss}')
                self.logger.debug('Evaluating model using validation set.')
                val_loss, self.val_acc_before = self.model.evaluate(self.validation_ds, verbose=self.verbose)
                self.logger.info(f'Val accuracy is {self.val_acc_before} and loss {val_loss}')
        else:

            self.logger.debug('Evaluating model using test set.')
            test_loss, self.test_acc_before = self.model.evaluate(self.test_ds, verbose=self.verbose)
            self.logger.info(f'Test accuracy is {self.test_acc_before} and loss {test_loss}')
            self.logger.debug('Evaluating model using validation set.')
            val_loss, self.val_acc_before = self.model.evaluate(self.validation_ds, verbose=self.verbose)
            self.logger.info(f'Val accuracy is {self.val_acc_before} and loss {val_loss}')

        self.logger.info('Finished environment initialization.')

    def get_highest_num_filters(self):
        filters = []
        previous_filters = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Flatten):
                break
            if isinstance(layer, tf.keras.layers.Conv2D):
                config = layer.get_config()
                if layer.name in self.original_layer_name_list:
                    filters.append(previous_filters)
                    filters.append(config['filters'])
                else:
                    previous_filters = config['filters']
            else:
                output = layer.output_shape
                if isinstance(output, list):
                    previous_filters = layer.output_shape[0][-1]
                else:
                    previous_filters = layer.output_shape[-1]

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

        if self.current_batch is None:
            if self.num_feature_maps>0:
                num_batches = self.num_feature_maps//self.tuning_batch_size

                self.logger.debug(f'For {self.num_feature_maps} samples, {num_batches} samples of {self.tuning_batch_size} are required.')

                if self.get_state_from == 'train':
                    self.current_batch = self.train_ds.take(num_batches)
                elif self.get_state_from == 'validation':
                    self.current_batch = self.validation_ds.take(num_batches)
                elif self.get_state_from == 'test':
                    self.current_batch = self.test_ds.take(num_batches)
                else:
                    raise "Please choose from 'train', 'validation' and 'test'."
            else:
                self.logger.debug('Using all feature maps')

                if self.get_state_from == 'train':
                    self.current_batch = self.train_ds
                elif self.get_state_from == 'validation':
                    self.current_batch = self.validation_ds
                elif self.get_state_from == 'test':
                    self.current_batch = self.test_ds
                else:
                    raise "Please choose from 'train', 'validation' and 'test'."

        self.logger.debug(f'Generating feature maps for layer {layer_name}')
        state = generate_fmp.predict(self.current_batch, verbose=self.verbose)

        del generate_fmp

        return state

    def get_layer_weights(self, layer_name):
        return self.model.get_layer(layer_name).get_weights()[0]

    def get_state(self, mode='current_state'):
        assert mode in ['current_state', 'next_state']
        if self._episode_ended:
            return None
        else:
            
            layer_name = None
            names = [layer.name for layer in self.model.layers]
            self.logger.debug('Layers are: {}'.format(names))
            retrieving = None
            if mode == 'current_state':
                retrieving = self.current_state_source
                if self.current_state_source == 'layer_input':
                    layer_idx = names.index(
                        self.layer_name_list[self._layer_counter]) - 1  
                elif self.current_state_source == 'layer_output':
                    layer_idx = names.index(
                        self.layer_name_list[self._layer_counter])
                elif self.current_state_source == 'layer_weights':
                    layer_name = self.layer_name_list[self._layer_counter]
            if mode == 'next_state':
                retrieving = self.next_state_source
                if self.next_state_source == 'layer_input':
                    layer_idx = names.index(
                        self.layer_name_list[self._layer_counter]) - 1
                elif self.next_state_source == 'layer_output':
                    layer_idx = names.index(
                        self.layer_name_list[self._layer_counter])
                elif self.next_state_source == 'layer_weights':
                    if self.layer_name_list[self._layer_counter] == self.layer_name_list[-1]:
                        layer_name = self.layer_name_list[self._layer_counter]
                    else:
                        layer_name = self.layer_name_list[self._layer_counter+1] 

            if layer_name is None:
                layer_name = names[layer_idx]
            self.logger.info(f'Getting {mode} - {retrieving} - {layer_name}.')

            if self.current_state_source == 'layer_weights':
                self.logger.debug(f'Getting weights of {layer_name}.')
                self._state = self.get_layer_weights(layer_name)
                # Reshape the kernel of a convolution into a rank 2 tensor (2D image without depth).
                if len(self._state.shape) == 4:
                    _, _, _, filters = self._state.shape
                    self._state = tf.reshape(self._state, shape=[-1, filters])

                # Add the depth to the image.
                self._state = np.expand_dims(self._state, -1)

                # Batch size equal to 1.
                self._state = np.expand_dims(self._state, 0)
                self.logger.debug(f'The state was reshaped into shape {self._state.shape}')
                
            else:
                self._state = self.get_output_feature_map(layer_name)
                self.logger.debug(f'{layer_name} has shape {self._state.shape}.')
                self.logger.debug(f'State is {self.current_state_source}. Getting output of layer {layer_name} and has shape {self._state.shape}.')
                if hasattr(self, 'conv_shape'):
                    if len(self._state.shape) == len(self.conv_shape):
                        b, h, w, c = self._state.shape
                        zeros = np.zeros(shape=(b,h,w,self.conv_shape[-1]))
                        zeros[:,:,:,:c] = self._state
                        self._state = zeros

                    elif len(self._state.shape) == len(self.dense_shape):
                        self._state = np.expand_dims(self._state, axis=-1)

            if np.isnan(self._state).any():
                self.logger.error('State array has NaN.')
                
            return self._state

    def reset(self):
        self.logger.debug('---RESTARTING ENVIRONMENT---')
        
        self.model = self.create_model_func()
        self.layer_name_list = self.original_layer_name_list.copy()
        self.callbacks = []
        self._layer_counter = 0
        self._episode_ended = False
        self.current_batch = None
        self._state = self.get_state('current_state')

        
        self.weights_before = int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))
        self.weights_previous_it = self.weights_before
        self.chosen_actions = []

        return self._state

    def calculate_metrics(self):
        pass

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        new_layers_it = []
        layer_name = self.layer_name_list[self._layer_counter]
        info = {'layer_name': layer_name}

        layer = self.model.get_layer(layer_name)
        if isinstance(layer, tf.keras.layers.Conv2D):
            compressors = self.conv_compressors
        else:
            compressors = self.dense_compressors

        self.logger.debug(f'Using action {action} on layer {layer_name}.')

        
        #Do nothing when no action or when all the dense layers will be removed despite being compressed.
        replace_dense_cond = compressors[action-1] == 'ReplaceDenseWithGlobalAvgPool' and self._layer_counter == len(self.layer_name_list)
        if replace_dense_cond:
            self.logger.debug('Setting action to 0 because ReplaceDense was not used in the first dense layer.')
        if action == 0 or replace_dense_cond:
            val_acc_after = self.val_acc_before
            val_loss = None
            test_loss = None
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
            new_layers_it.append(compressor.new_layer_name)


        if compressors[action] == 'ReplaceDenseWithGlobalAvgPool':
            self._state = self.get_state('next_state')
            self._episode_ended = True
            new_layers_it.append(self.model.layers[-1].name)
            self.layer_name_list.append(self.model.layers[-1].name)
        else:
            self._state = self.get_state('next_state')
            self._layer_counter += 1


        if self._layer_counter == len(self.layer_name_list):
            self._episode_ended = True


        if self.tuning_mode == 'layer':
                train_layers = new_layers_it
        else:
            train_layers = self.layer_name_list


        if (self.tuning_mode == 'layer' or self._episode_ended) and train_layers:
            rbw = RestoreBestWeights(acc_before = self.val_acc_before, reward_func=self.reward_func, weights_before=self.weights_before, verbose=1)
            temp_cb = copy.copy(self.callbacks)
            temp_cb.append(rbw)


            self.logger.debug(f'Only {train_layers} are trainable.')
            if self.strategy:
                self.logger.debug('Strategy found. Using strategy to fit model.')

                # Extract core info of model to create another inside scope.
                layers, configs, weights = extract_model_parts(self.model)

                with self.strategy.scope():
                    optimizer2 = tf.keras.optimizers.Adam(1e-5)
                    loss2 = tf.keras.losses.SparseCategoricalCrossentropy()
                    metric2 = tf.keras.metrics.SparseCategoricalAccuracy()
                    self.model = create_model_from_parts(layers, configs, weights, optimizer2, loss2, metric2, input_shape=self.input_shape)
                    for layer in self.model.layers:
                        if layer.name in train_layers:
                            layer.trainable = True
                        else:
                            layer.trainable = False
                    # self.model.summary()

                    self.model.fit(self.train_ds, epochs=self.tuning_epochs, callbacks=temp_cb, validation_data=self.validation_ds, verbose=self.verbose)
                    test_loss, test_acc_after = self.model.evaluate(self.test_ds, verbose=self.verbose)
                    val_loss, val_acc_after = self.model.evaluate(self.validation_ds, verbose=self.verbose)
                
                # Set all layers back to trainable.
                for layer in self.model.layers:
                    layer.trainable = True

            else:

                # Train only the modified layers.
                for layer in self.model.layers:
                    if layer.name in train_layers:
                        layer.trainable = True
                    else:
                        layer.trainable = False
                # self.model.summary()
                self.model.fit(self.train_ds, epochs=self.tuning_epochs, callbacks=temp_cb, validation_data=self.validation_ds, verbose=self.verbose)

                # Set all layers back to trainable.
                for layer in self.model.layers:
                    layer.trainable = True

                test_loss, test_acc_after = self.model.evaluate(self.test_ds, verbose=self.verbose)
                val_loss, val_acc_after = self.model.evaluate(self.validation_ds, verbose=self.verbose)
        else:
            test_acc_after = self.test_acc_before
            val_acc_after = self.val_acc_before
            weights_after = self.weights_previous_it

        self.logger.info(f'Val loss: {val_loss}\t Val acc:{val_acc_after}')
        self.logger.info(f'Test loss: {test_loss}\t Test acc:{test_acc_after}')
        weights_after = calculate_model_weights(self.model)

        stats = {'weights_before': self.weights_previous_it, 
                 'weights_after': weights_after, 
                 'accuracy_before': self.test_acc_before,
                 'accuracy_after': test_acc_after}
        
        reward_step = self.reward_func(stats)
        stats['weights_before'] = self.weights_before
        reward_all_steps = self.reward_func(stats)
        # reward_step = 1 - (weights_after / self.weights_previous_it) + test_acc_after - 0.9 * self.test_acc_before
        # reward_all_steps = 1 - (weights_after / self.weights_before) + test_acc_after - 0.9 * self.test_acc_before
        self.weights_previous_it = weights_after

        self.logger.info(f'Reward for step is {reward_step}. The reward for all steps is {reward_all_steps}.')

        info['test_acc_before'] = self.test_acc_before
        info['test_acc_after'] = test_acc_after
        info['weights_original'] = self.weights_before
        info['weights_before_step'] = self.weights_previous_it
        info['weights_after'] = weights_after
        info['val_acc_before'] = self.val_acc_before
        info['val_acc_after'] = val_acc_after
        info['actions'] = self.chosen_actions
        info['reward_step'] = reward_step
        info['reward_all_steps'] = reward_all_steps
        
        

        return self._state, reward_all_steps, self._episode_ended, info

class ModelCompressionSVDEnv(ModelCompressionEnv):
    def __init__(self,*args, **kwargs):
        (super(ModelCompressionSVDEnv, self).__init__)(*args,**kwargs)

    def action_space(self):
        return {'low':0.01, 'high':1.0}

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        new_layers_it = []
        layer_name = self.layer_name_list[self._layer_counter]
        info = {'layer_name': layer_name}

        layer = self.model.get_layer(layer_name)
        if isinstance(layer, tf.keras.layers.Conv2D):
            compressors = self.conv_compressors
        else:
            compressors = self.dense_compressors

        self.logger.debug(f'Using action {action} on layer {layer_name}.')

        weights_before = self.weights_previous_it
        if action == 1.0:
            val_acc_after = self.val_acc_before
            test_acc_after = self.test_acc_before
            self.logger.debug(f'Layer {layer_name} was not compressed.')
            self.chosen_actions.append(action)

        else:
            self.logger.debug(f'Compressing layer {layer_name} using {compressors[0]} and value {action}.')

            # Save the sequence of actions.
            self.chosen_actions.append(action)

            
            class_ = getattr(CompressionTechniques, compressors[0])

            compressor = class_(model=self.model, dataset=self.train_ds, optimizer=self.optimizer, loss=self.loss_object, metrics=self.train_metric, fine_tuning=False, input_shape=self.input_shape, tuning_verbose=self.verbose, callbacks=self.callbacks)

            compressor.callbacks = self.callbacks

            self.compr_params[compressors[0]]['layer_name'] = layer_name
            self.compr_params[compressors[0]]['percentage'] = action

            self.logger.debug('Params: {}'.format(self.compr_params[compressors[0]]))
            compressor.compress_layer(**self.compr_params[compressors[0]])

            # Get compressed model
            self.model = compressor.get_model()

            self.callbacks = compressor.callbacks
            self.layer_name_list[self._layer_counter] = compressor.new_layer_name
            new_layers_it.append(compressor.new_layer_name)


        self._state = self.get_state('next_state')
        self._layer_counter += 1
            


        if self._layer_counter == len(self.layer_name_list):
            self.logger.debug('Episode ended.')
            self._episode_ended = True
            
        if self.tuning_epochs>0 and (self.tuning_mode =='layer' or self._episode_ended): 
            self.logger.debug(f'Fine-tuning the model for {self.tuning_epochs} epochs in mode {self.tuning_mode}.')
            if self.strategy:
                # Extract core info of model to create another inside scope.
                layers, configs, weights = extract_model_parts(self.model)
                with self.strategy.scope():
                    optimizer2 = tf.keras.optimizers.Adam(1e-5)
                    loss2 = tf.keras.losses.SparseCategoricalCrossentropy()
                    metric2 = tf.keras.metrics.SparseCategoricalAccuracy()
                    self.model = create_model_from_parts(layers, configs, weights, optimizer2, loss2, metric2, input_shape=self.input_shape)

                    # for layer in self.model.layers:
                    #     if layer.name == layer_name:
                    #         layer.trainable = True
                    #     else:
                    #         layer.trainable = False
                    rbw = RestoreBestWeights(acc_before = self.val_acc_before, weights_before=self.weights_before, verbose=1)
                    self.model.fit(self.train_ds, epochs=self.tuning_epochs, callbacks=[rbw], validation_data=self.validation_ds, verbose=self.verbose)
                    # for layer in self.model.layers:
                    #     layer.trainable = True
            else:
                # for layer in self.model.layers:
                #     if layer.name == layer_name:
                #         layer.trainable = True
                #     else:
                #         layer.trainable = False
                rbw = RestoreBestWeights(acc_before = self.val_acc_before, weights_before=self.weights_before, verbose=1)
                self.model.fit(self.train_ds, epochs=self.tuning_epochs, callbacks=[rbw], validation_data=self.validation_ds, verbose=self.verbose)
                # for layer in self.model.layers:
                #     layer.trainable = True
            

        if self.strategy:
            self.logger.debug('Strategy found. Using strategy to evaluate model.')

            # Extract core info of model to create another inside scope.
            layers, configs, weights = extract_model_parts(self.model)

            with self.strategy.scope():
                optimizer2 = tf.keras.optimizers.Adam(1e-5)
                loss2 = tf.keras.losses.SparseCategoricalCrossentropy()
                metric2 = tf.keras.metrics.SparseCategoricalAccuracy()
                self.model = create_model_from_parts(layers, configs, weights, optimizer2, loss2, metric2, input_shape=self.input_shape)
                test_loss, test_acc_after = self.model.evaluate(self.test_ds, verbose=self.verbose)
                val_loss, val_acc_after = self.model.evaluate(self.validation_ds, verbose=self.verbose)
        else:
            if action < 1.0:
                test_loss, test_acc_after = self.model.evaluate(self.test_ds, verbose=self.verbose)
                val_loss, val_acc_after = self.model.evaluate(self.validation_ds, verbose=self.verbose)
            else:
                test_acc_after = self.test_acc_before
                val_acc_after = self.val_acc_before

        weights_after = calculate_model_weights(self.model)
        reward_step = 1 - (weights_after / weights_before) + test_acc_after - 0.9 * self.test_acc_before
        reward_all_steps = 1 - (weights_after / self.weights_before) + test_acc_after - 0.9 * self.test_acc_before
        self.weights_previous_it = weights_after

        info['test_acc_before'] = self.test_acc_before
        info['test_acc_after'] = test_acc_after
        info['weights_original'] = self.weights_before
        info['weights_before_step'] = weights_before
        info['weights_after'] = weights_after
        info['val_acc_before'] = self.val_acc_before
        info['val_acc_after'] = val_acc_after
        info['actions'] = self.chosen_actions
        info['reward_step'] = reward_step
        info['reward_all_steps'] = reward_all_steps
        
        
        return self._state, reward_all_steps, self._episode_ended, info


class ModelCompressionSVDIntEnv(ModelCompressionEnv):
    def __init__(self,*args, **kwargs):
        (super(ModelCompressionSVDIntEnv, self).__init__)(*args,**kwargs)
        self.possible_actions = [5] + list(range(10,110,10))

    def action_space(self):
        return self.possible_actions

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        new_layers_it = []
        layer_name = self.layer_name_list[self._layer_counter]
        info = {'layer_name': layer_name}

        layer = self.model.get_layer(layer_name)
        if isinstance(layer, tf.keras.layers.Conv2D):
            compressors = self.conv_compressors
        else:
            compressors = self.dense_compressors

        self.logger.debug(f'Using action {action} on layer {layer_name}.')

        weights_before = self.weights_previous_it
        action = self.possible_actions[action]
        if action == 100 :
            val_acc_after = self.val_acc_before
            test_acc_after = self.test_acc_before
            self.logger.debug(f'Layer {layer_name} was not compressed.')
            self.chosen_actions.append(action)

        else:
            self.logger.debug(f'Compressing layer {layer_name} using {compressors[0]} and value {action}.')

            # Save the sequence of actions.
            self.chosen_actions.append(action)

            
            class_ = getattr(CompressionTechniques, compressors[0])

            compressor = class_(model=self.model, dataset=self.train_ds, optimizer=self.optimizer, loss=self.loss_object, metrics=self.train_metric, fine_tuning=False, input_shape=self.input_shape, tuning_verbose=self.verbose, callbacks=self.callbacks)

            compressor.callbacks = self.callbacks

            self.compr_params[compressors[0]]['layer_name'] = layer_name
            self.compr_params[compressors[0]]['percentage'] = action

            self.logger.debug('Params: {}'.format(self.compr_params[compressors[0]]))
            compressor.compress_layer(**self.compr_params[compressors[0]])

            # Get compressed model
            self.model = compressor.get_model()

            self.callbacks = compressor.callbacks
            self.layer_name_list[self._layer_counter] = compressor.new_layer_name
            new_layers_it.append(compressor.new_layer_name)


        self._state = self.get_state('next_state')
        self._layer_counter += 1
            


        if self._layer_counter == len(self.layer_name_list):
            self.logger.debug('Episode ended.')
            self._episode_ended = True
            
        if self.tuning_epochs>0 and (self.tuning_mode =='layer' or self._episode_ended): 
            self.logger.debug(f'Fine-tuning the model for {self.tuning_epochs} epochs in mode {self.tuning_mode}.')
            if self.strategy:
                # Extract core info of model to create another inside scope.
                layers, configs, weights = extract_model_parts(self.model)
                with self.strategy.scope():
                    optimizer2 = tf.keras.optimizers.Adam(1e-5)
                    loss2 = tf.keras.losses.SparseCategoricalCrossentropy()
                    metric2 = tf.keras.metrics.SparseCategoricalAccuracy()
                    self.model = create_model_from_parts(layers, configs, weights, optimizer2, loss2, metric2, input_shape=self.input_shape)

                    # for layer in self.model.layers:
                    #     if layer.name == layer_name:
                    #         layer.trainable = True
                    #     else:
                    #         layer.trainable = False
                    rbw = RestoreBestWeights(acc_before = self.val_acc_before, reward_func=self.reward_func,weights_before=self.weights_before, verbose=1)
                    self.model.fit(self.train_ds, epochs=self.tuning_epochs, callbacks=[rbw], validation_data=self.validation_ds, verbose=self.verbose)
                    # for layer in self.model.layers:
                    #     layer.trainable = True
            else:
                # for layer in self.model.layers:
                #     if layer.name == layer_name:
                #         layer.trainable = True
                #     else:
                #         layer.trainable = False
                rbw = RestoreBestWeights(acc_before = self.val_acc_before, reward_func=self, weights_before=self.weights_before, verbose=1)
                self.model.fit(self.train_ds, epochs=self.tuning_epochs, callbacks=[rbw], validation_data=self.validation_ds, verbose=self.verbose)
                # for layer in self.model.layers:
                #     layer.trainable = True
            

        if self.strategy:
            self.logger.debug('Strategy found. Using strategy to evaluate model.')

            # Extract core info of model to create another inside scope.
            layers, configs, weights = extract_model_parts(self.model)

            with self.strategy.scope():
                optimizer2 = tf.keras.optimizers.Adam(1e-5)
                loss2 = tf.keras.losses.SparseCategoricalCrossentropy()
                metric2 = tf.keras.metrics.SparseCategoricalAccuracy()
                self.model = create_model_from_parts(layers, configs, weights, optimizer2, loss2, metric2, input_shape=self.input_shape)
                test_loss, test_acc_after = self.model.evaluate(self.test_ds, verbose=self.verbose)
                val_loss, val_acc_after = self.model.evaluate(self.validation_ds, verbose=self.verbose)
        else:
            if action != 0:
                test_loss, test_acc_after = self.model.evaluate(self.test_ds, verbose=self.verbose)
                val_loss, val_acc_after = self.model.evaluate(self.validation_ds, verbose=self.verbose)
            else:
                test_acc_after = self.test_acc_before
                val_acc_after = self.val_acc_before

        weights_after = calculate_model_weights(self.model)

 
        if self._episode_ended:
            stats = {'weights_before': self.weights_before, 'weights_after':weights_after, 'accuracy_after': test_acc_after}
            reward = self.reward_func(stats)
        else: 
            reward = 0
        
        self.weights_previous_it = weights_after

        info['test_acc_before'] = self.test_acc_before
        info['test_acc_after'] = test_acc_after
        info['val_acc_before'] = self.val_acc_before
        info['val_acc_after'] = val_acc_after
        info['weights_original'] = self.weights_before
        info['weights_before_step'] = weights_before
        info['weights_after'] = weights_after
        info['actions'] = self.chosen_actions
        info['reward'] = reward
        
        
        return self._state, reward, self._episode_ended, info






