import tensorflow as tf
import logging
import numpy as np
from CompressionLibrary import utils

class AddSparseConnectionsCallback(tf.keras.callbacks.Callback):

    def __init__(self, layer_name, target_perc=0.75, conn_perc_per_epoch=0.1):
        super(AddSparseConnectionsCallback, self).__init__()
        self.target_perc = target_perc
        self.conn_perc_per_epoch = conn_perc_per_epoch
        self.layer_name = layer_name
        self.logger = logging.getLogger(__name__)

    def on_epoch_end(self, epoch, logs=None):
        self.logger.debug(f'Updating sparse connections of layer {self.layer_name}.')
        names = [layer.name for layer in self.model.layers]
        layer_idx = names.index(self.layer_name)
        connections = np.asarray(self.model.layers[layer_idx].get_connections())
        total_connections = connections.shape[0]
        num_connections = np.sum(connections)
        perc_connections = num_connections / total_connections
        if perc_connections < self.target_perc:
            select_n = int(total_connections * self.conn_perc_per_epoch)
            remaining_to_goal = int(total_connections * (self.target_perc - perc_connections))
            smallest = min(select_n, remaining_to_goal)
            missing_connections = np.argwhere(connections==0).flatten()
            self.logger.debug(f'Adding {smallest} connections of {missing_connections.shape[0]} missing.')
            new_connections = np.random.choice(missing_connections, smallest,
              replace=False)
            connections[new_connections] = 1
            self.logger.debug(f'Number of activated connections {np.sum(connections)} of {connections.shape[0]}.')
            self.model.layers[layer_idx].set_connections(connections.tolist())

class EarlyStoppingReward(tf.keras.callbacks.Callback):

    def __init__(self, weights_before, baseline_acc=0.3, min_delta=0, patience=5, verbose=0, restore_best_weights=True):
        """
        
        """
        self.patience = patience
        self.verbose = verbose
        self.weights_after = None
        self.weights_before = weights_before
        self.baseline_acc = baseline_acc
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.logger = logging.getLogger(__name__)
        
        self.monitor_op = np.greater
        self.min_delta *= 1
       

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best_acc = -np.Inf
        self.best_reward = -np.Inf
        self.best_weights = None
        self.best_epoch = 0
            

    def on_epoch_end(self, epoch, logs=None):
        current_acc, current_reward = self.get_monitor_value(logs)

        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current_acc, self.best_acc):
            self.best_acc = current_acc
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self._is_improvement(current_acc, self.baseline_acc):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    self.logger.info(
                        f'Restoring model weights from the end of the best epoch: {self.best_epoch + 1}.')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            self.logger.info(
                f'Epoch {self.stopped_epoch + 1}: early stopping with {self.best} {self.monitor}.')


    def get_monitor_values(self, logs):
        logs = logs or {}
        self.logger.debug(f'Old model had {self.weights_before} weights.')
        acc_after = logs.get('val_sparse_categorical_accuracy')
        weights_after = utils.calculate_model_weights(self.model)
        self.logger.debug(f'Model has {acc_after} accuracy, {weights_after} weights.')
        reward = 1 - (weights_after / self.weights_before) + acc_after
        self.logger.debug(f'Reward is {reward}.')

        return acc_after, reward

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)