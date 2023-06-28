import tensorflow as tf
from CompressionLibrary.custom_layers import ROIEmbedding, ROIEmbedding1D
from CompressionLibrary.utils import OUActionNoise
import numpy as np
import logging
from functools import partial


class Agent:
    def __init__(self, name, state_shape, n_actions, epsilon=0.9, layer_type='fc'):
        """Base class for an agent."""
        self.name = name
        
        self.model = self.model_creation(name, state_shape, n_actions, layer_type)
        self.logger = logging.getLogger(__name__)


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
    def __init__(self, epsilon, *args, **kwargs):
        self.epsilon = epsilon
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


class DuelingDQNAgent(Agent):
    def __init__(self, epsilon, *args, **kwargs):
        self.epsilon = epsilon
        (super(DuelingDQNAgent, self).__init__)(*args,**kwargs)

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
          x = tf.keras.layers.Conv2D(64, kernel_size=3)(input)
          x = tf.keras.layers.Conv2D(64, kernel_size=3)(x)
          x = ROIEmbedding(n_bins=[(4,4), (2,2), (1,1)])(x)
          v = tf.keras.layers.Dense(512, activation='linear')(x)
          v = tf.keras.layers.Dense(1, activation='linear')(v)
          a = tf.keras.layers.Dense(512, activation='linear')(x)
          a = tf.keras.layers.Dense(n_actions, activation='linear')(a)
          output = tf.keras.layers.Lambda(lambda inputs: inputs[0] + (inputs[1] - tf.math.reduce_mean(inputs[1], axis=1, keepdims=True)))([v,a])
        model = tf.keras.Model(inputs=input, outputs=output, name=name)
        return model

    def get_qvalues(self, state):
        qvalues = self.model(state)
        return qvalues

    def sample_actions(self, qvalues, exploration=True):

        if exploration:
            _, n_actions = qvalues.shape
            random_action = np.random.choice(n_actions, size=1)[0]
            best_actions = qvalues.argmax(axis=-1)
            actions, counts = np.unique(best_actions, return_counts=True)
            index = np.argmax(counts)
            best_action = actions[index]
            action = np.random.choice([best_action, random_action], size=1, p=[1 - self.epsilon, self.epsilon])
            return action
        else:
            best_actions = qvalues.argmax(axis=-1)
            actions, counts = np.unique(best_actions, return_counts=True)
            index = np.argmax(counts)
            best_action = actions[index]
        
            return [best_action]

class DDPGWeights2D(Agent):
    def __init__(self, *args, **kwargs):
        (super(DDPGWeights2D, self).__init__)( *args, **kwargs)
        std_dev = 0.2
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        # self.noise = partial(np.random.normal, loc=0.0, scale=std_dev)
        self.min_value = 0.01
        self.max_value = 1.0

    def get_shared_layers(self, state_shape):
        shared_input = tf.keras.layers.Input(shape=(None, None, state_shape[-1]))
        x = tf.keras.layers.Conv2D(128, kernel_size=3)(shared_input)
        x = tf.keras.layers.Conv2D(128, kernel_size=3)(x)
        shared_output = ROIEmbedding(n_bins=[(8,8),(4,4), (2,2), (1,1)])(x)
        return shared_input, shared_output
        
    def get_actor(self, shared_input, shared_output):
        x = tf.keras.layers.Dense(256, activation='relu')(shared_output)
        x = tf.keras.layers.Dense(256, activation='relu')(x)       
        output = tf.keras.layers.Dense(1, activation='sigmoid')(shared_output)
        self.actor = tf.keras.Model(inputs=shared_input, outputs=output, name='actor')

    def get_critic(self, shared_input, shared_output):
        action_input = tf.keras.layers.Input(shape=(1))
        action_output = tf.keras.layers.Dense(256, activation='relu')(action_input)

        x = tf.keras.layers.Dense(256, activation='relu')(shared_output)
        x = tf.keras.layers.Dense(256, activation='relu')(x) 
        x = tf.keras.layers.Concatenate()([x, action_output])
        output = tf.keras.layers.Dense(1)(x)
        self.critic = tf.keras.Model(inputs=[shared_input, action_input], outputs=output, name='critic')

    def model_creation(self, name, state_shape, n_actions, layer_type):
        shared_input, shared_output = self.get_shared_layers(state_shape)
        self.get_actor(shared_input, shared_output)
        self.get_critic(shared_input, shared_output)

    def policy(self, state, exploration=False):
        sampled_actions = self.actor(state)
        sampled_actions = np.mean(sampled_actions)
        self.logger.debug(f'Average action before legalizing is {sampled_actions}.')
        legal_action = np.clip(sampled_actions, self.min_value, self.max_value)
        self.logger.debug(f'Action is {legal_action}.')
        if exploration:
            noise = self.noise()[0]
            self.logger.debug(f'Noise: {noise}.')
            legal_action = legal_action + noise
            self.logger.debug(f'Action after noise: {legal_action}.')
            legal_action = np.clip(legal_action, self.min_value, self.max_value)
            self.logger.debug(f'Action after clipping: {legal_action}.')

        return legal_action

class DDPG(Agent):
    def __init__(self, min_value, max_value, *args, **kwargs):
        (super(DDPG, self).__init__)( *args, **kwargs)
        std_dev = 0.2
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        self.min_value = min_value
        self.max_value = max_value

    def get_shared_layers(self, state_shape, layer_type):
        if layer_type=='fc':
            inputs = tf.keras.layers.Input(shape=(None, 1))
            x = tf.keras.layers.Conv1D(128, kernel_size=3)(inputs)
            x = tf.keras.layers.Conv1D(128, kernel_size=3)(x)
            shared_output = ROIEmbedding1D(n_bins=[64, 32, 16, 8 ,4, 2, 1])(x)
          
        else:          
            inputs = tf.keras.layers.Input(shape=(None, None, state_shape[-1]))
            x = tf.keras.layers.Conv2D(128, kernel_size=3)(inputs)
            x = tf.keras.layers.Conv2D(128, kernel_size=3)(x)
            shared_output = ROIEmbedding(n_bins=[(8,8),(4,4), (2,2), (1,1)])(x)
            
        return inputs, shared_output

    def get_actor(self, shared_input, shared_output):
        x = tf.keras.layers.Dense(256, activation='relu')(shared_output)
        x = tf.keras.layers.Dense(256, activation='relu')(x)       
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.actor = tf.keras.Model(inputs=shared_input, outputs=output)

    def get_critic(self,  n_actions, shared_input, shared_output):
        action_input = tf.keras.layers.Input(shape=(n_actions))
        x = tf.keras.layers.Concatenate()([shared_output, action_input])
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        output = tf.keras.layers.Dense(1)(x)
       
        self.critic = tf.keras.Model(inputs=[shared_input, action_input], outputs=output)


    def model_creation(self, name, state_shape, n_actions, layer_type):
        shared_input, shared_output = self.get_shared_layers(state_shape, layer_type)
        self.get_actor(shared_input, shared_output)
        self.get_critic(n_actions, shared_input, shared_output)

    def policy(self, state, exploration=False):
        sampled_actions = self.actor(state)
        mean = np.mean(sampled_actions)
        self.logger.debug(f'Average action before legalizing is {mean}.')
        legal_action = np.clip(sampled_actions, self.min_value, self.max_value)[0]
        legal_action = np.mean(legal_action)
        self.logger.debug(f'Action is {legal_action}.')
        if exploration:
            legal_action = legal_action + self.noise()
            self.logger.debug(f'Action after noise: {legal_action}.')
            legal_action = np.clip(legal_action, self.min_value, self.max_value)[0]
            self.logger.debug(f'Action after clipping: {legal_action}.')

        

        return legal_action