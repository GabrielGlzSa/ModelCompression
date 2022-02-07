import abc
import tensorflow as tf
from tf_agents.environments import tf_environment, py_environment
from tf_agents.specs import array_spec
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as ts
import numpy as np

from utils import load_dataset

import CompressionTechniques


class LayerOpPyEnv(py_environment.PyEnvironment):
    def __init__(self, compressors, model_path, compr_params,
                 dataset, validation,
                 layer_name):
        self.compressors = compressors
        self.model_path = model_path
        self.dataset = dataset
        self.validation = validation
        self.layer_name = layer_name
        self.compr_params = compr_params
        self.model = tf.keras.models.load_model(model_path, compile=True)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(compressors), name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.model.get_layer(layer_name).kernel.shape, dtype=np.int32, name='observation'
        )
        kernel = self.model.get_layer(layer_name).kernel
        print(kernel)
        self._state = self.model.get_layer(layer_name).kernel.shape
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self.model.get_layer(self.layer_name).kernel.shape,
        self._episode_ended = False
        self.model =tf.keras.models.load_model(self.model_path, compile=True)
        return ts.restart(self._state, dtype=np.int32)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        class_ = getattr(CompressionTechniques, self.compressors[action])
        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        loss, val_acc_before = self.model.evaluate(self.validation)
        compressor = class_(model=self.model, dataset=self.dataset, optimizer=optimizer, loss=loss_object, metrics=train_metric,
                            fine_tune=True)


        if self.compressors[action] in self.compr_params.keys():
            self.compr_params[self.compressors[action]]['layer_name'] = self.layer_name
            compressor.compress_layer(**self.compr_params[self.compressors[action]])
        else:
            compressor.compress_layer(layer_name=self.layer_name)

        self.model = compressor.get_model()
        weights_before, weights_after = compressor.get_weights_diff()
        loss, val_acc_after = self.model.evaluate(self.validation)

        reward = (weights_after / weights_before) + val_acc_after

        return ts.termination(np.array(self.model.get_layer(self.layer_name).kernel.shape), reward)



if __name__=='__main__':
    train_ds, valid_ds, test_ds, input_shape, num_classes = load_dataset('mnist')
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_0',
                                                        input_shape=input_shape),
                                 tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
                                 tf.keras.layers.MaxPool2D((2, 2), 2),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(128, activation='relu', name='dense_0'),
                                 tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
                                 tf.keras.layers.Dense(num_classes, activation='softmax', name='dense_softmax')
                                 ])
    model.compile(optimizer=optimizer, loss=loss_object, metrics=train_metric)
    model.fit(train_ds, epochs=5, validation_data=valid_ds)


    model_path = './data/full_model/test'
    model.save(model_path)

    compressors_cnn = ['DepthwiseSeparableConvolution', 'FireLayerCompression', 'InsertSVDConv', 'SparseConnectionsCompression']
    compressors_fcn = ['DeepCompression', 'InsertDenseSVD', 'InsertDenseSVDCustom', 'InsertDenseSparse', 'ReplaceDenseWithGlobalAvgPool']

    parameters = {}
    parameters['DeepCompression'] = {'layer_name': 'dense_0', 'threshold': 0.001}
    parameters['ReplaceDenseWithGlobalAvgPool'] = {'layer_name': 'dense_1'}
    parameters['InsertDenseSVD'] = {'layer_name': 'dense_0', 'units': 16}
    parameters['InsertDenseSVDCustom'] = {'layer_name': 'dense_0', 'units': 16}
    parameters['InsertDenseSparse'] = {'layer_name': 'dense_0', 'verbose': True, 'units': 16}
    parameters['InsertSVDConv'] = {'layer_name': 'conv2d_1', 'units': 8}
    parameters['DepthwiseSeparableConvolution'] = {'layer_name': 'conv2d_1'}
    parameters['FireLayerCompression'] = {'layer_name': 'conv2d_1'}
    parameters['MLPCompression'] = {'layer_name': 'conv2d_1'}
    parameters['SparseConnectionsCompression'] = {'layer_name': 'conv2d_1', 'epochs': 20,
                                                  'target_perc': 0.75, 'conn_perc_per_epoch': 0.1}

    layer_name = "conv2d_1"
    env = LayerOpPyEnv(compressors_cnn, model_path, parameters,
                 train_ds, valid_ds,
                 layer_name)
    utils.validate_py_environment(env, episodes=1)
