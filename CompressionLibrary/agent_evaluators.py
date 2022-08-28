from asyncio.log import logger
import tensorflow as tf
from CompressionLibrary.environments import ModelCompressionEnv
import pandas as pd
import numpy as np
import gc
import time
import logging

def make_env_imagenet(create_model, train_ds, valid_ds, test_ds, input_shape, layer_name_list, num_feature_maps, tuning_batch_size, current_state_source='layer_input', next_state_source='layer_output', strategy=None):
 
    

    w_comprs = ['InsertDenseSVD', 'InsertDenseSparse',
                'DeepCompression'] 
    c_comprs = ['InsertSVDConv', 'SparseConvolutionCompression',
                'DepthwiseSeparableConvolution', 'SparseConnectionsCompression']
    l_comprs = ['FireLayerCompression', 'MLPCompression',
                'ReplaceDenseWithGlobalAvgPool']
    compressors_list = w_comprs + c_comprs + l_comprs

    parameters = {}
    parameters['DeepCompression'] = {
        'layer_name': None, 'threshold': 0.001}
    parameters['ReplaceDenseWithGlobalAvgPool'] = {'layer_name': None}
    parameters['InsertDenseSVD'] = {'layer_name': None}
    parameters['InsertDenseSparse'] = {'layer_name': None,  'new_layer_iterations':2000, 'new_layer_verbose':True}
    parameters['InsertSVDConv'] = {'layer_name': None}
    parameters['DepthwiseSeparableConvolution'] = {'layer_name': None}
    parameters['FireLayerCompression'] = {'layer_name': None}
    parameters['MLPCompression'] = {'layer_name': None}
    parameters['SparseConvolutionCompression'] = {
        'layer_name': None, 
        'new_layer_iterations': 1000,
        'new_layer_iterations_sparse':3000,
        'new_layer_verbose':True} 
    parameters['SparseConnectionsCompression'] = {'layer_name': None, 
                                                  'target_perc': 0.75, 'conn_perc_per_epoch': 0.15}

    env = ModelCompressionEnv(compressors_list, create_model, parameters,
                                      train_ds, valid_ds, test_ds,
                                      layer_name_list, input_shape, current_state_source=current_state_source, next_state_source=next_state_source, 
                                      num_feature_maps=num_feature_maps, tuning_batch_size=tuning_batch_size, verbose=1, strategy=strategy)

    return env


def play_and_record(conv_agent, fc_agent, env, conv_replay, fc_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time
    """
    # initial state
    s = env.reset()
    # Play the game for n_steps as per instructions above

    logger = logging.getLogger(__name__)
    data = []
    rewards = []
    for it in range(n_steps):
        for k in range(1, len(env.layer_name_list)+1):
            tf.keras.backend.clear_session()
            # Get the current layer name
            current_layer_name = env.layer_name_list[env._layer_counter]
            # Get the layer.
            layer = env.model.get_layer(current_layer_name)

            was_conv = True
            # Choose agent depending on layer type.
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Calculate q values for batch of images
                qvalues = conv_agent.get_qvalues(s).numpy()
                # Action is the mode of the action.
                action = conv_agent.sample_actions_using_mode(qvalues)[0]
            if isinstance(layer, tf.keras.layers.Dense):
                was_conv = False
                # Calculate q values for batch of images
                qvalues = fc_agent.get_qvalues(s).numpy()
                # Action is the mode of the action.
                action = fc_agent.sample_actions_using_mode(qvalues)[0]

            # Apply action
            new_s, r, done, info = env.step(action)


            logger.debug(f'Iteration {it} - Layer {current_layer_name} {k}/{len(env.layer_name_list)}\tChosen action {action} has {r} reward.')
            logger.debug(info)

            row = {'state': s, 'action': action, 'reward': r,
                   'next_state': new_s, 'done': done, 'info': info, 'was_conv':was_conv}
            
            data.append(row)

            s = env.get_state('current_state')

            if done:
                s = env.reset()
                break
            gc.collect()

        # Correct reward is the last value of r.
        rewards.append(r)

        for row in data:
            if row['was_conv']:
                for idx, state in enumerate(row['state']):
                    conv_replay.add(state, row['action'], rewards[-1], row['next_state'][idx], row['done'])
            else:
                for idx, state in enumerate(row['state']):
                    fc_replay.add(state, row['action'], rewards[-1], row['next_state'][idx], row['done'])

        del data

    return np.mean(rewards)

def evaluate_agents(env, conv_agent, fc_agent, save_name, n_games=2):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    acc = []
    weights = []
    infos = []
    df_results = pd.DataFrame()
    logger = logging.getLogger(__name__)
    total_time = 0
    for game_id in range(n_games):
        tf.keras.backend.clear_session()
        s = env.reset()
        info = None
        start = time.time()
        for k in range(1,len(env.layer_name_list)+1):
            next_layer_name = env.layer_name_list[env._layer_counter]
            layer = env.model.get_layer(next_layer_name)
            if isinstance(layer, tf.keras.layers.Conv2D):
                qvalues = conv_agent.get_qvalues(s).numpy()
                action = conv_agent.sample_actions_using_mode(qvalues)[0]
            if isinstance(layer, tf.keras.layers.Dense):
                qvalues = fc_agent.get_qvalues(s).numpy()
                action = fc_agent.sample_actions_using_mode(qvalues)[0]

            _ , r, done, info = env.step(action)
            s = env.get_state('current_state')
            logger.debug(f'Game {game_id} - Layer {next_layer_name} {k}/{len(env.layer_name_list)}\tChosen action {action}.')
            logger.debug(info)

            if done:
                s = env.reset()
                break
        game_time = start - time.time()
        actions = info['actions']
        logger.info(f'Actions taken in game {game_id} were  {actions} for a reward of {r}. Took {game_time} seconds.')
        total_time += game_time

        df_results = df_results.append(info, ignore_index=True)
        # Calculate reward using stats before and after compression
        reward = df_results.iloc[-1]['reward']

        rewards.append(reward)
        acc.append(info['test_acc_after'])
        weights.append(info['weights_after'])
        infos.append(info['actions'])
    df_results.to_csv(save_name, index=False)
    logger.info('Evaluation of {n_games} took {total_time} secs.')

    return np.mean(rewards), np.mean(acc), np.mean(weights)