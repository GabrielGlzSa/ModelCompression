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

    def __init__(self, monitor='val_reward', min_delta=0, mode='max',patience=3, verbose=0, acc_before=None, weights_before=None, baseline=None, restore_best_weights=True):
        """
        
        """
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.acc_before = acc_before
        self.weights_after = None
        self.weights_before = weights_before
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.logger = logging.getLogger(__name__)
        

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if (self.monitor.endswith('acc') or self.monitor.endswith('accuracy') or
                self.monitor.endswith('auc') or self.monitor.endswith('reward')):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0
        if self.weights_before is None:
            self.weights_before = utils.calculate_model_weights(self.model)

        if self.monitor == 'val_reward' and self.baseline is None:
            self.baseline = self.acc_before
            

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
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
                f'Epoch {self.stopped_epoch + 1}: early stopping with {self.best} {self.monitor}')

    def get_model_reward(self, logs):
        self.logger.debug(f'Old model had {self.weights_before} weights.')
        acc_after = logs.get('val_sparse_categorical_accuracy')
        weights_after = utils.calculate_model_weights(self.model)
        self.logger.debug(f'Model has {acc_after} accuracy, {weights_after} weights.')
        reward = 1 - (weights_after / self.weights_before) + acc_after
        self.logger.debug(f'Reward is {reward}.')
        
        return reward

    def get_monitor_value(self, logs):
        logs = logs or {}
        if self.monitor == 'val_reward':
            monitor_value = self.get_model_reward(logs)
        else:
            monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            self.logger.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)