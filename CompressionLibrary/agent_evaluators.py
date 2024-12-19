from asyncio.log import logger
import tensorflow as tf
from CompressionLibrary.environments import *
import pandas as pd
import numpy as np
import gc
from datetime import datetime
import logging
import sys



def make_env_adadeep(create_model, reward_func, train_ds, valid_ds, test_ds, input_shape, layer_name_list, num_feature_maps, tuning_batch_size, tuning_epochs, verbose=0, get_state_from='train',tuning_mode='layer', current_state_source='layer_input', next_state_source='layer_output', strategy=None):

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
    parameters['InsertDenseSVD'] = {'layer_name': None, 'percentage':None}
    parameters['InsertDenseSparse'] = {'layer_name': None,  'new_layer_iterations':2000, 'new_layer_verbose':True}
    parameters['InsertSVDConv'] = {'layer_name': None}
    parameters['DepthwiseSeparableConvolution'] = {'layer_name': None}
    parameters['FireLayerCompression'] = {'layer_name': None}
    parameters['MLPCompression'] = {'layer_name': None, 'percentage':None}
    parameters['SparseConvolutionCompression'] = {
        'layer_name': None, 
        'new_layer_iterations': 1000,
        'new_layer_iterations_sparse':3000,
        'new_layer_verbose':True} 
    parameters['SparseConnectionsCompression'] = {'layer_name': None, 
                                                  'target_perc': 0.75, 'conn_perc_per_epoch': 0.15}

    env = EnvDiscreteUniqueActions(compressors_list=compressors_list, 
                                    create_model_func=create_model, 
                                    compr_params=parameters, 
                                    train_ds=train_ds, 
                                    validation_ds=valid_ds, 
                                    test_ds=test_ds,
                                    state_ds=test_ds, 
                                    layer_name_list=layer_name_list, 
                                    input_shape=input_shape, 
                                    reward_func=reward_func,
                                    tuning_batch_size=tuning_batch_size, 
                                    tuning_epochs=tuning_epochs, 
                                    tuning_mode=tuning_mode, 
                                    current_state_source=current_state_source,
                                    next_state_source=next_state_source, 
                                    num_feature_maps=num_feature_maps, 
                                    verbose=verbose,
                                    strategy=strategy)
    return env


def play_and_record(conv_agent, fc_agent, env, conv_replay, fc_replay, run_id, test_number, dataset_name, save_name, n_games=1):
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
    rewards = []
    try:
        df_results = pd.read_csv(save_name)
    except:
        df_results = pd.DataFrame()

    for it in range(n_games):
        start = datetime.now()
        data = []
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
                if np.isnan(qvalues).any():
                    logger.error('Qvalues have NaN.')
                    sys.exit()
                logger.debug(f'Q values for conv2d layer  are {qvalues}')
                # Action is the mode of the action.
                action = conv_agent.sample_actions_exploration(qvalues)[0]
                logger.debug(f'E-Greedy action for conv2d layer is {action}')
            if isinstance(layer, tf.keras.layers.Dense):
                was_conv = False
                # Calculate q values for batch of images
                qvalues = fc_agent.get_qvalues(s).numpy()
                if np.isnan(qvalues).any():
                    logger.error('Qvalues have NaN.')
                    sys.exit()
                logger.debug(f'Q values for dense layer are {qvalues}')
                # Action is the mode of the action.
                action = fc_agent.sample_actions_exploration(qvalues)[0]
                logger.debug(f'E-Greedy action for dense layer is {action}')

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

        
        actions = info['actions']
        info['actions'] = ','.join(actions)
        info['run_id'] = run_id
        info['test_number'] = test_number
        info['game_id'] = it
        info['dataset'] = dataset_name
        del info['layer_name']
        new_row = pd.DataFrame(info, index=[0])
        df_results = pd.concat([df_results, new_row], ignore_index=True)

        # Correct reward is the last value of r.
        rewards.append(r)
        end = datetime.now()
        time_diff = (end - start).total_seconds()
        logger.info(f'Took {time_diff} seconds for one compression.')
        for row in data:
            if row['was_conv']:
                for idx, state in enumerate(row['state']):
                    conv_replay.add(state, row['action'], rewards[-1], row['next_state'][idx], row['done'], dataset_name)
            else:
                for idx, state in enumerate(row['state']):
                    fc_replay.add(state, row['action'], rewards[-1], row['next_state'][idx], row['done'], dataset_name)

        del data

    df_results.to_csv(save_name, index=False)

    return np.mean(rewards)

def evaluate_agents(env, conv_agent, fc_agent, run_id, test_number, dataset_name,save_name, n_games=2, agent_name=None):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    acc = []
    weights = []
    infos = []
    logger = logging.getLogger(__name__)
    total_time = 0
    try:
        df_results = pd.read_csv(save_name)
    except:
        df_results = pd.DataFrame()

    for game_id in range(n_games):
        tf.keras.backend.clear_session()
        s = env.reset()
        start = datetime.now()
        for k in range(1,len(env.layer_name_list)+1):
            next_layer_name = env.layer_name_list[env._layer_counter]
            layer = env.model.get_layer(next_layer_name)
            if isinstance(layer, tf.keras.layers.Conv2D):
                qvalues = conv_agent.get_qvalues(s).numpy()
                action = conv_agent.sample_actions_greedy(qvalues)[0]
                logger.debug(f'Greedy action for conv2d layer is {action}')
            if isinstance(layer, tf.keras.layers.Dense):
                qvalues = fc_agent.get_qvalues(s).numpy()
                action = fc_agent.sample_actions_greedy(qvalues)[0]
                logger.debug(f'Greedy action for dense layer is {action}')

            _ , r, done, info = env.step(action)
            s = env.get_state('current_state')
            logger.debug(f'Game {game_id} - Layer {next_layer_name} {k}/{len(env.layer_name_list)}\tChosen action {action}.')
            logger.debug(info)

            if done:
                s = env.reset()
                break
        game_time = (datetime.now() - start).total_seconds()
        actions = info['actions']
        info['agent_name'] = agent_name
        info['actions'] = ','.join(actions)
        info['run_id'] = run_id
        info['test_number'] = test_number
        info['game_id'] = game_id
        info['dataset'] = dataset_name
        
        del info['layer_name']
        logger.info(f'Actions taken in game {game_id} were  {actions} for a reward of {r}. Took {game_time} seconds.')
        total_time += game_time
        new_row = pd.DataFrame(info, index=[0])
        df_results = pd.concat([df_results, new_row], ignore_index=True)

        reward = info['reward_all_steps']

        rewards.append(reward)
        acc.append(info['test_acc_after'])
        weights.append(info['weights_after'])
        infos.append(info['actions'])
        df_results.to_csv(save_name, index=False)
    logger.info(f'Evaluation of {n_games} took {total_time} secs. An average of {total_time/n_games} secs per game.')

    return np.mean(rewards), np.mean(acc), np.mean(weights)