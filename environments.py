import tensorflow as tf
import numpy as np
import inspect
import importlib
from utils import load_dataset
import CompressionTechniques
from CompressionTechniques import *

class LayerEnv():
    def __init__(self, compressors_list, model_path, compr_params,
                 dataset, validation,
                 layer_name_list):

        self.model_path = model_path
        self.dataset = dataset
        self.validation = validation
        self.layer_name_list = layer_name_list
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
            temp_comp = class_(model=self.model, dataset=self.dataset, optimizer=self.optimizer, loss=self.loss_object, metrics=self.train_metric,
                            fine_tune=True)
            if temp_comp.target_layer_type == 'conv':
                self.conv_compressors.append(compressor)
            elif temp_comp.target_layer_type == 'dense':
                self.dense_compressors.append(compressor)

        self.model = tf.keras.models.load_model(model_path, compile=True)
        self._state = self.model.get_layer(self.layer_name_list[0]).get_weights()[0].shape
        self._layer_counter = 0

        self._episode_ended = False

    def observation_space(self):
        return len(self._state)

    def action_space(self):
        layer = self.model.get_layer(self.layer_name_list[self._layer_counter])
        if isinstance(layer, tf.keras.layers.Conv2D):
            return len(self.conv_compressors) + 1
        elif isinstance(layer, tf.keras.layers.Dense):
            return len(self.dense_compressors) + 1
        else:
            raise "Unsupported type of layer. Only Conv2D and Dense layers are supported."

    def reset(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=True)
        self._state = self.model.get_layer(self.layer_name_list[0]).get_weights()[0].shape
        self._layer_counter = 0
        self._episode_ended = False

        return self._state

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        info = {'action_overwritten': False}
        layer_name = self.layer_name_list[self._layer_counter]
        layer = self.model.get_layer(layer_name)
        if isinstance(layer, tf.keras.layers.Conv2D):
            compressors = self.conv_compressors
        else:
            compressors = self.dense_compressors

        if action > len(compressors) + 1:
            action = 0
            info['action_overwritten'] = True

        # loss, val_acc_before = self.model.evaluate(self.validation)

        if action == 0:
            weight_diff = 0
        else:
            print('action_before:', action )
            action -= 1
            print(compressors)
            print(compressors[action])


            class_ = getattr(CompressionTechniques, compressors[action])
            compressor = class_(model=self.model, dataset=self.dataset, optimizer=self.optimizer, loss=self.loss_object, metrics=self.train_metric,
                                fine_tune=True)
            if compressors[action] in self.compr_params.keys():
                self.compr_params[compressors[action]]['layer_name'] = layer_name
                compressor.compress_layer(**self.compr_params[compressors[action]])
            else:
                compressor.compress_layer(layer_name=layer_name)

            self.model = compressor.get_model()

            weights_before, weights_after = compressor.get_weights_diff()
            weight_diff = weights_after / weights_before

        loss, val_acc_after = self.model.evaluate(self.validation)

        reward = weight_diff + val_acc_after

        if compressors[action] == 'ReplaceDenseWithGlobalAvgPool':
            self._episode_ended = True
            self._state = self.model.layers[-1].get_weights()[0].shape
        else:
            self._layer_counter +=1
            print('counter is now ', self._layer_counter, len(self.layer_name_list))
            if self._layer_counter == len(self.layer_name_list):
                self._episode_ended = True
            else:
                self._state = self.model.get_layer(self.layer_name_list[self._layer_counter]).get_weights()[0].shape

        return self._state, reward, self._episode_ended, info
