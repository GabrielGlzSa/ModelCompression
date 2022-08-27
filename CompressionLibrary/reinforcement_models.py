import tensorflow as tf
from CompressionLibrary.custom_layers import ROIEmbedding, ROIEmbedding1D
import numpy as np


class DQNAgent:
    def __init__(self, name, state_shape, n_actions, epsilon=0.0, layer_type='fc'):
        """A simple DQN agent"""
        self.name = name
        
        if layer_type=='fc':
          input = tf.keras.layers.Input(shape=(None, state_shape[-1]))
          x = ROIEmbedding1D(n_bins=[32, 16, 8 ,4, 2, 1])(input)
          x = tf.keras.layers.Dense(512, activation='relu')(x)
          output = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.linear)(x)
        else:          
          input = tf.keras.layers.Input(shape=(None, None, state_shape[-1]))
          x = tf.keras.layers.Conv2D(512, kernel_size=3)(input) #64
          x = tf.keras.layers.Conv2D(512, kernel_size=3)(x) #64
          x = ROIEmbedding(n_bins=[(4,4), (2,2), (1,1)])(x)
          x = tf.keras.layers.Dense(512, activation='relu')(x)
          output = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.linear)(x)
        self.model = tf.keras.Model(inputs=input, outputs=output, name=name)
        self.epsilon = epsilon

    def get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues = self.model(state_t)
        return qvalues

    def get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        qvalues = self.model(state_t)
        return qvalues

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

    def sample_actions_using_mode(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_action = np.random.choice(n_actions, size=1)[0]
        best_actions = qvalues.argmax(axis=-1)
        actions, counts = np.unique(best_actions, return_counts=True)
        index = np.argmax(counts)
        best_action = actions[index]
        action = np.random.choice([best_action, random_action], size=1, p=[1 - epsilon, epsilon])
        return action



class RandomAgent:
    def __init__(self, name, n_actions):
        """A simple DQN agent"""
        self.name = name
        self.n_actions = n_actions

    def get_qvalues(self, state_t):
        """Return dummy Q-values that will not be used."""
        return tf.zeros(shape=[self.n_actions], dtype='float32')

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        return np.random.choice(range(self.n_actions), size=qvalues.shape[0], replace=True)

    def sample_actions_using_mode(self, qvalues):
        """pick random action. """
        random_action = np.random.choice(range(self.n_actions), size=1)
        return random_action