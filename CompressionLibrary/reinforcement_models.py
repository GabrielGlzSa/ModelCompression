import tensorflow as tf
from CompressionLibrary.custom_layers import ROIEmbedding, ROIEmbedding1D
from CompressionLibrary.utils import OUActionNoise
import numpy as np


class Agent:
    def __init__(self, name, state_shape, n_actions, epsilon, layer_type='fc'):
        """Base class for an agent."""
        self.name = name
        self.epsilon = epsilon
        self.model = self.model_creation(name, state_shape, n_actions, layer_type)

    def get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        qvalues = self.model(state_t)
        return qvalues

    def sample_actions_exploration(self, qvalues):
        """Picks an action for exploration."""
        _, n_actions = qvalues.shape
        random_action = np.random.choice(n_actions, size=1)[0]
        best_actions = qvalues.argmax(axis=-1)
        actions, counts = np.unique(best_actions, return_counts=True)
        index = np.argmax(counts)
        best_action = actions[index]
        action = np.random.choice([best_action, random_action], size=1, p=[1 - self.epsilon, self.epsilon])
        return action

    def sample_actions_greedy(self, qvalues):
        """Choose greedy action. """
        best_actions = qvalues.argmax(axis=-1)
        actions, counts = np.unique(best_actions, return_counts=True)
        index = np.argmax(counts)
        best_action = actions[index]
    
        return [best_action]


class RandomAgent(Agent):
    def __init__(self, **kwargs):
        (super(RandomAgent, self).__init__)(**kwargs)

    def model_creation(self, **kwargs):
        return None

    def get_qvalues(self, state_t):
        """Return dummy Q-values that will not be used."""
        return tf.zeros(shape=[self.n_actions], dtype='float32')

    def sample_actions_exploration(self, qvalues):
        """Picks an action for exploration."""
        return np.random.choice(range(self.n_actions), size=qvalues.shape[0], replace=True)

    def sample_actions_greedy(self, qvalues):
        """Choose greedy action. """
        random_action = np.random.choice(range(self.n_actions), size=1)
        return random_action

class DQNAgent(Agent):
    def __init__(self, **kwargs):
        (super(DQNAgent, self).__init__)(**kwargs)

    def model_creation(self,name, state_shape, n_actions, layer_type):
        if layer_type=='fc':
          input = tf.keras.layers.Input(shape=(None, 1))
          x = tf.keras.layers.Conv1D(64, kernel_size=3)(input)
          x = ROIEmbedding1D(n_bins=[32, 16, 8 ,4, 2, 1])(x)
          x = tf.keras.layers.Dense(512, activation='relu')(x)
          output = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.linear)(x)
        else:          
          input = tf.keras.layers.Input(shape=(None, None, state_shape[-1]))
          x = tf.keras.layers.Conv2D(64, kernel_size=3)(input) #64
          x = tf.keras.layers.Conv2D(64, kernel_size=3)(x) #64
          x = ROIEmbedding(n_bins=[(4,4), (2,2), (1,1)])(x)
          x = tf.keras.layers.Dense(512, activation='relu')(x)
          output = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.linear)(x)
        model = tf.keras.Model(inputs=input, outputs=output, name=name)
        return model

class DuelingDQNAgent(Agent):
    def __init__(self, **kwargs):
        (super(DuelingDQNAgent, self).__init__)(**kwargs)

    def model_creation(self, name, state_shape, n_actions, layer_type):
        if layer_type=='fc':
          input = tf.keras.layers.Input(shape=(None, 1))
          x = tf.keras.layers.Conv1D(64, kernel_size=3)(input)
          x = ROIEmbedding1D(n_bins=[32, 16, 8 ,4, 2, 1])(x)
          v = tf.keras.layers.Dense(512, activation='linear')(x)
          v = tf.keras.layers.Dense(1, activation='linear')(v)
          a = tf.keras.layers.Dense(512, activation='linear')(x)
          a = tf.keras.layers.Dense(n_actions, activation='linear')(a)
          output = tf.keras.layers.Lambda(lambda inputs: inputs[0] + (inputs[1] - tf.math.reduce_mean(inputs[1], axis=1, keepdims=True)))([v,a])
        else:          
          input = tf.keras.layers.Input(shape=(None, None, state_shape[-1]))
          x = tf.keras.layers.Conv2D(64, kernel_size=3)(input) #64
          x = tf.keras.layers.Conv2D(64, kernel_size=3)(x) #64
          x = ROIEmbedding(n_bins=[(4,4), (2,2), (1,1)])(x)
          v = tf.keras.layers.Dense(512, activation='linear')(x)
          v = tf.keras.layers.Dense(1, activation='linear')(v)
          a = tf.keras.layers.Dense(512, activation='linear')(x)
          a = tf.keras.layers.Dense(n_actions, activation='linear')(a)
          output = tf.keras.layers.Lambda(lambda inputs: inputs[0] + (inputs[1] - tf.math.reduce_mean(inputs[1], axis=1, keepdims=True)))([v,a])
        model = tf.keras.Model(inputs=input, outputs=output, name=name)
        return model

class DDPG:
    def __init__(self, **kwargs):
        (super(Agent, self).__init__)(**kwargs)
        std_dev = 0.2
        ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))


    def get_actor(state_shape, layer_type):
        if layer_type=='fc':
            input = tf.keras.layers.Input(shape=(None, 1))
            x = tf.keras.layers.Conv1D(64, kernel_size=3)(input)
            x = tf.keras.layers.Conv1D(64, kernel_size=3)(x)
            x = ROIEmbedding1D(n_bins=[32, 16, 8 ,4, 2, 1])(x)

        else:          
            input = tf.keras.layers.Input(shape=(None, None, state_shape[-1]))
            x = tf.keras.layers.Conv2D(64, kernel_size=3)(input)
            x = tf.keras.layers.Conv2D(128, kernel_size=3)(x)
            x = ROIEmbedding(n_bins=[(4,4), (2,2), (1,1)])(x)

        x = tf.keras.layers.Dense(512, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')
        model = tf.keras.Model(inputs=input, outputs=output)

    def get_critic(self, state_shape, n_actions, layer_type):
        if layer_type=='fc':
            state_input = tf.keras.layers.Input(shape=(None, 1))
            x = tf.keras.layers.Conv1D(64, kernel_size=3)(state_input)
            x = tf.keras.layers.Conv1D(64, kernel_size=3)(x)
            x = ROIEmbedding1D(n_bins=[32, 16, 8 ,4, 2, 1])(x)
            state_output = tf.keras.layers.Dense(512, activation='relu')(x)

            action_input = tf.keras.layers.Input(shape=(n_actions))
            action_output = tf.keras.layers.Dense(512, activation='relu')(action_input)


        else:          
            input = tf.keras.layers.Input(shape=(None, None, state_shape[-1]))
            x = tf.keras.layers.Conv2D(64, kernel_size=3)(input)
            x = tf.keras.layers.Conv2D(128, kernel_size=3)(x)
            x = ROIEmbedding(n_bins=[(4,4), (2,2), (1,1)])(x)
            state_output = tf.keras.layers.Dense(512, activation='relu')(x)

            action_input = tf.keras.layers.Input(shape=(n_actions))
            action_output = tf.keras.layers.Dense(512, activation='relu')(action_input)

          

        concat = tf.keras.layers.Concatenate()([state_output, action_output])
        output = tf.keras.layers.Dense(256, activation='relu')(concat)
        output = tf.keras.layers.Dense(256, activation='relu')(output)
        output = tf.keras.layers.Dense(1)(output)
        self.critic = tf.keras.Model(inputs=[state_input, action_input], outputs=output)

    def model_creation(self, name, state_shape, n_actions, layer_type):
        self.get_actor(state_shape, layer_type)
        self.get_actor(state_shape, n_actions, layer_type)

        return None

    def policy(self, state, exploration=False):
        sampled_actions = tf.squeeze(self.actor(state))
        sampled_actions = sampled_actions.numpy() + self.noise
        legal_action = np.clip(sampled_actions, 0.1, 1.0)
        return [np.squeeze(legal_action)]


    def sample_actions_exploration(self, state):
        """Picks an action for exploration."""
        actions = self.policy(state, True)
        action = np.mean(actions)
        return action

    def sample_actions_greedy(self, state):
        """Choose greedy action. """
        actions = self.policy(state, False)
        action = np.mean(actions)
        return action
