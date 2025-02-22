import tensorflow as tf
import logging
import numpy as np
from CompressionLibrary import utils
import pandas as pd
import os

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

class RestoreBestWeights(tf.keras.callbacks.Callback):

    def __init__(self, acc_before, weights_before, reward_func, dataset_name, save_name, verbose=0, stop_when_better=False):
        """
        
        """
        self.verbose = verbose
        self.weights_after = None
        self.weights_before = weights_before
        self.reward_func = reward_func
        self.acc_before = acc_before
        self.dataset_name = dataset_name
        self.best_weights = None
        self.stop_when_better = stop_when_better
        self.logger = logging.getLogger(__name__)
        
        self.monitor_op = np.greater
        self.stats = {'weights_before': self.weights_before,
                      'weights_after':None, 
                      'accuracy_before':self.acc_before,
                      'accuracy_after': None}
        
        self.save_name = save_name
        
       

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.best_acc = - np.inf
        self.best_weights = self.model.get_weights()
        weights_after = utils.calculate_model_weights(self.model)
        self.stats['weights_after'] = weights_after
        self.stats['accuracy_after'] = self.stats['accuracy_before']
        self.best_reward = self.reward_func(self.stats)
        self.best_epoch = 0
            

    def on_epoch_end(self, epoch, logs=None):
        current_acc, current_reward, weights_after = self.get_monitor_values(logs)
        self.logger.info(f'Epoch {epoch} has a reward of {current_reward}.')
        info = {
            'dataset': self.dataset_name,
            'epoch': epoch,
            'weights_before': self.weights_before,
            'weights_after': weights_after,
            'acc_before': self.acc_before,
            'acc_after': current_acc,
            'rw': current_reward
            }
        
        new_row = pd.DataFrame(info, index=[0])
        if not os.path.isfile(self.save_name):
            new_row.to_csv(self.save_name, index=False)
        else: # else it exists so append without writing the header
            new_row.to_csv(self.save_name, mode='a', index=False, header=False)

        if self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        losses = [logs.get('loss'), logs.get('val_loss')]
        if np.isnan(losses).any():
            self.logger.warning('Loss is NaN, reverting weights to preven NaN.')
            self.model.set_weights(self.best_weights)
            self.model.stop_training = True
        else:
            if self._is_improvement(current_acc, self.best_acc):
                    self.logger.info(f'Saving weights due to {current_acc} being better than {self.best_acc}.')
                    self.best_acc = current_acc
                    self.best_reward = current_reward
                    self.best_epoch = epoch
                    self.best_weights = self.model.get_weights()

        if self.stop_when_better and current_acc > self.acc_before:
            self.model.stop_training = True
            
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            self.logger.info(
                f'Restoring weights of epoch {self.best_epoch} due to achieving {self.best_acc} val accuracy and {self.best_reward} reward.')
        self.model.set_weights(self.best_weights)    

    def get_monitor_values(self, logs):
        logs = logs or {}
        self.logger.info(f'Old model had {self.weights_before} weights.')
        acc_after = logs.get('val_sparse_categorical_accuracy')
        weights_after = utils.calculate_model_weights(self.model)
        self.logger.info(f'Model has {acc_after} accuracy, {weights_after} weights.')

        self.stats['weights_after'] = weights_after
        self.stats['accuracy_after'] = acc_after
        reward = self.reward_func(self.stats)        

        return acc_after, reward, weights_after

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value, reference_value)