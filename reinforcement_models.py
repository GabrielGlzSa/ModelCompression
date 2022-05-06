import tensorflow as tf
from custom_layers import ROIEmbedding, ROIEmbedding1D
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
          x = tf.keras.layers.Conv2D(64, kernel_size=3)(input)
          x = tf.keras.layers.Conv2D(64, kernel_size=3)(x)
          x = ROIEmbedding(n_bins=[(4,4), (2,2), (1,1)])(x)
          x = tf.keras.layers.Dense(512, activation='relu')(x)
          output = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.linear)(x)
        self.model = tf.keras.Model(inputs=input, outputs=output, name=name)
        self.model.summary()
        self.weights = self.model.trainable_weights
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
