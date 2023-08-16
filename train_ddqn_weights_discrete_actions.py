
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_datasets as tfds
import logging

from CompressionLibrary.environments import ModelCompressionSVDIntEnv
from CompressionLibrary.reinforcement_models import DuelingDQNAgent
from CompressionLibrary.replay_buffer import PrioritizedExperienceReplayBufferMultipleDatasets
from CompressionLibrary.utils import calculate_model_weights
from CompressionLibrary.reward_functions import reward_MnasNet_penalty as calculate_reward

from uuid import uuid4
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
import gc


dataset_names = ['mnist']#['fashion_mnist','kmnist','mnist']
agent_name = 'DDQN_weights_discrete_tuning_zero_rw_' + '-'.join(dataset_names)
run_id = datetime.now().strftime('%Y-%m-%d-%H-%M%S-') + str(uuid4())

try:
  # Use below for TPU
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
  tf.config.experimental_connect_to_cluster(resolver)
  # This is the TPU initialization code that has to be at the beginning.
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("All devices: ", tf.config.list_logical_devices('TPU'))
  strategy = tf.distribute.TPUStrategy(resolver)
  data_path = '/mnt/disks/mcdata/data'

except:
  print('ERROR: Not connected to a TPU runtime; Using GPU strategy instead!')
  strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
  data_path = './data'
  
if strategy:
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(f'/home/A00806415/DCC/ModelCompression/data/logs/ModelCompression_{agent_name}.log', 'w+')],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

log = logging.getLogger('tensorflow')
log.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

exploration_filename = data_path + f'/stats/{agent_name}_training.csv'
test_filename = data_path + f'/stats/{agent_name}_testing.csv'
agents_path = data_path+f'/agents/DDQN/checkpoints/LeNet_{agent_name}'
figures_path = data_path+f'/figures/{agent_name}'


# Parameters shared in training and testing env
current_state = 'layer_weights'
next_state = 'layer_weights'
tuning_epochs = 0
tuning_mode = 'layer'

batch_size_per_replica = 128
tuning_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync


# Env variables
training_state_set_source = 'validation'
training_num_feature_maps = 256


# Testing variables
testing_state_set_source = 'validation'
testing_num_feature_maps = -1
eval_n_samples = 1
test_frequency_epochs = 20 # Test every 20 epochs.


# Replay variables
replay_buffer_size = 1000000

# RL training variables

verbose = 0
rl_iterations = 4000
update_weights_iterations = 10
rl_batch_size = 128
gamma = 0.99
beta = 0.5
learning_rate = 1e-5
epsilon_start_value = 1.0
min_epsylon = 0.1
epsylon_decay = 0.9997





layer_name_list = ['conv2d_1',  'dense', 'dense_1']

def create_model(dataset_name, train_ds, valid_ds):
    checkpoint_path = f"./data/models/lenet_{dataset_name}/cp.ckpt"
    optimizer = tf.keras.optimizers.Adam(1e-5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    input = tf.keras.layers.Input((28,28,1))
    x = tf.keras.layers.Conv2D(6, (5,5), padding='SAME', activation='sigmoid', name='conv2d')(input)
    x = tf.keras.layers.AveragePooling2D((2,2), strides=2, name='avg_pool_1')(x)
    x = tf.keras.layers.Conv2D(16, (5,5), padding='VALID', activation='sigmoid', name='conv2d_1')(x)
    x = tf.keras.layers.AveragePooling2D((2,2), strides=2, name='avg_pool_2')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(120, activation='sigmoid', name='dense')(x)
    x = tf.keras.layers.Dense(84, activation='sigmoid', name='dense_1')(x)
    x = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(input, x, name='LeNet')
    model.compile(optimizer=optimizer, loss=loss_object,
                    metrics=[train_metric])

    try:
        model.load_weights(checkpoint_path).expect_partial()
    except:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)
        model.fit(train_ds,
          epochs=3000,
          validation_data=valid_ds,
          callbacks=[cp_callback])

    return model       

def dataset_preprocessing(img, label):
    img = tf.cast(img, tf.float32)
    img = img/255.0
    return img, label

def load_dataset(dataset_name, batch_size=128):
    splits, info = tfds.load(dataset_name, as_supervised=True, with_info=True, shuffle_files=True, 
                                split=['train[:80%]', 'train[80%:]','test'])

    (train_examples, validation_examples, test_examples) = splits
    num_examples = info.splits['train'].num_examples

    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape

    input_shape = (28,28,1)

    train_ds = train_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = validation_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_examples.map(dataset_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, test_ds, input_shape, num_classes



input_shape = (28,28,1)

def create_environments(dataset_names, num_feature_maps, state_set_source):
    w_comprs = ['InsertDenseSVD'] 
    l_comprs = ['MLPCompression']
    compressors_list = w_comprs +  l_comprs

    parameters = {}
    parameters['InsertDenseSVD'] = {'layer_name': None, 'percentage': None}
    parameters['MLPCompression'] = {'layer_name': None, 'percentage': None}
    environments = []
    for dataset in dataset_names:
        train_ds, valid_ds, test_ds, input_shape, _ = load_dataset(dataset, tuning_batch_size)
        new_func = partial(create_model, dataset_name=dataset, train_ds=train_ds, valid_ds=valid_ds)
        env = ModelCompressionSVDIntEnv(
                reward_func=calculate_reward,
                compressors_list=compressors_list, 
                create_model_func=new_func, 
                compr_params=parameters, 
                train_ds=train_ds, 
                validation_ds=valid_ds, 
                test_ds=test_ds, 
                layer_name_list=layer_name_list, 
                input_shape=input_shape, 
                tuning_batch_size=tuning_batch_size, 
                tuning_epochs=tuning_epochs,
                get_state_from=state_set_source, 
                current_state_source=current_state, 
                next_state_source=next_state, 
                num_feature_maps=num_feature_maps, 
                verbose=verbose,
                strategy=strategy)

        environments.append(env)

    return environments

envs = create_environments(dataset_names,num_feature_maps=training_num_feature_maps, state_set_source=training_state_set_source)
test_envs = create_environments(dataset_names,num_feature_maps=testing_num_feature_maps, state_set_source=testing_state_set_source)

conv_shape, dense_shape = envs[0].observation_space()
action_space = envs[0].action_space()
num_actions = len(action_space)

print(conv_shape, dense_shape)


fc_n_actions = conv_n_actions = num_actions

print(f'The action space is {action_space}')


with strategy.scope(): 

    conv_agent = DuelingDQNAgent(
        name="ddqn_agent_conv", state_shape=conv_shape, n_actions=conv_n_actions, epsilon=epsilon_start_value, layer_type='cnn')
    conv_target_network = DuelingDQNAgent(
        name="target_conv", state_shape=conv_shape, n_actions=conv_n_actions, epsilon=epsilon_start_value,layer_type='cnn')

    conv_agent.model.summary()
    try:
        conv_agent.model.load_weights(agents_path+'_conv.cpkt')
        conv_target_network.model.load_weights(agents_path+'_conv_target.ckpt')
    except:
        print('Failed to find pretrained models for the RL agents.')
        pass


    optimizer_conv = tf.keras.optimizers.Adam(learning_rate)

@tf.function
def update_agent_conv(state_batch, action_batch, reward_batch, next_state_batch, done, sample_probabilities):
    agent_best_actions = tf.cast(tf.math.argmax(conv_agent.get_qvalues(next_state_batch), axis=1), tf.int32)
    indices = tf.stack([tf.range(state_batch.shape[0]), agent_best_actions], axis=1)
    target_agent_qvalues = tf.gather_nd(conv_target_network.get_qvalues(next_state_batch), indices=indices)
    reference_qvalues = reward_batch + gamma * target_agent_qvalues * (1.0 - done)

    masks = tf.one_hot(action_batch, num_actions)
    with tf.GradientTape() as tape:
        q_values = conv_agent.get_qvalues(state_batch)
        selected_action_qvalues = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        td_error = tf.abs(reference_qvalues - selected_action_qvalues)
        deltai = td_error
        importance_sampling = 1 / (replay_buffer_size** beta * sample_probabilities**beta)
        td_loss = tf.math.reduce_mean((importance_sampling*deltai))


    gradients = tape.gradient(td_loss, conv_agent.model.trainable_weights)
    optimizer_conv.apply_gradients(zip(gradients, conv_agent.model.trainable_weights))
    return td_error

original_weights = np.mean([calculate_model_weights(env.model) for env in envs])
original_acc = np.mean([env.test_acc_before for env in envs])

mean_weights_history = [original_weights]
mean_acc_history = [original_acc]
mean_rw_history = [0]
td_loss_history_conv = []


def sample_batch(exp_replay, batch_size, highest_td_error):
    obs_batch, act_batch, reward_batch, next_obs_batch, done, probs, td_indexes = exp_replay.sample(
        batch_size, highest_td_error=highest_td_error)
    return {
        'state_batch': tf.cast(obs_batch.to_tensor(), tf.float32),
        'action_batch': tf.cast(act_batch, tf.int32),
        'reward_batch': tf.cast(reward_batch, tf.float32),
        'next_state_batch': tf.cast(next_obs_batch.to_tensor(), tf.float32),
        'done': tf.cast(done, tf.float32),
        'sample_probabilities': tf.cast(probs, tf.float32),
        'td_indexes': td_indexes
    }
    

def calculate_td_error(agent, target_agent, state_batch, action_batch, reward_batch, next_state_batch, done):
    agent_best_actions = tf.cast(tf.math.argmax(agent.get_qvalues(next_state_batch), axis=1), tf.int32)
    logger.debug(f"Action for S: {tf.reduce_mean(action_batch)}. Action for S': {tf.math.reduce_mean(agent_best_actions)}.")
    indices = tf.stack([tf.range(state_batch.shape[0]), agent_best_actions], axis=1)
    target_agent_qvalues = tf.gather_nd(target_agent.get_qvalues(next_state_batch), indices=indices)
    reference_qvalues = reward_batch + gamma * target_agent_qvalues * (1.0 - done)
    masks = tf.one_hot(action_batch, num_actions)
    q_values = agent.get_qvalues(state_batch)
    logger.debug(f"R + gamma * Q'(S') - Q(S)")
    selected_action_qvalues = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
    td_error = (reference_qvalues - selected_action_qvalues)
    logger.debug(f'{tf.reduce_mean(reward_batch)} + {gamma} * {tf.reduce_mean(target_agent_qvalues * (1.0 - done))} - {tf.reduce_mean(selected_action_qvalues)} = {tf.reduce_mean(td_error)}')
    return td_error

def play_and_record(conv_agent, env, conv_replay, run_id, test_number, dataset_name, save_name, n_games=10, exploration=True):
    # initial state
    s = env.reset()
    # Play the game for n_steps as per instructions above

    logger = logging.getLogger(__name__)
    rewards = []
    acc = []
    weights = []
    infos = []
    total_time = 0

    for it in range(n_games):
        start = datetime.now()
        last_conv_data = None
        skip_add_replay = False
        data = []
        for k in range(1, len(env.layer_name_list)+1):
            tf.keras.backend.clear_session()
            # Get the current layer name
            current_layer_name = env.layer_name_list[env._layer_counter]
            # Get the layer.
            layer = env.model.get_layer(current_layer_name)


            # Calculate q values for batch of images
            qvalues = conv_agent.get_qvalues(s)
            action = conv_agent.sample_actions(qvalues.numpy(), exploration=exploration)[0]
           
            # Action is the mode of the action.
            
            logger.debug(f'Action for layer {current_layer_name} layer is {action}')

            # Apply action
            new_s, r, done, info = env.step(action) 

            logger.debug(f'Iteration {it} - Layer {current_layer_name} {k}/{len(env.layer_name_list)}\tChosen action {action} has {r} reward.')
            logger.debug(info)

            num_inst = s.shape[0]

            if exploration:
                # data.append([s, action, r, new_s, done])
                done_float = 1.0 if done else 0.0
                num_inst = s.shape[0]
                actions_batch = np.array([action]*num_inst)
                logger.debug(f'Conv replay has {len(conv_replay)} examples.')
                td_errors = calculate_td_error(conv_agent, conv_target_network, s, actions_batch, [r]*num_inst, new_s, done_float )
                td_errors = np.reshape(np.abs(td_errors), -1)
                conv_replay.add_multiple(s, [action]*num_inst, [r]*num_inst, new_s, td_errors, [done]*num_inst, dataset_name)
                logger.debug(f'Conv replay has {len(conv_replay)} examples.')
                logging.debug(f'Layer TD error is {td_errors}')


            s = env.get_state('current_state')

            if done:                        
                s = env.reset()
                break

        gc.collect()
        

        # Using 0f as actions are percentages without decimals.
        info['actions'] = ','.join(['{:.0f}'.format(x) for x in info['actions']] )
        info['run_id'] = run_id
        info['test_number'] = test_number
        info['game_id'] = it
        info['dataset'] = dataset_name
        del info['layer_name']
        rewards.append(r)
        acc.append(info['test_acc_after'])
        weights.append(info['weights_after'])
        new_row = pd.DataFrame(info, index=[0])
        if not os.path.isfile(save_name):
            new_row.to_csv(save_name, index=False)
        else: # else it exists so append without writing the header
            new_row.to_csv(save_name, mode='a', index=False, header=False)

        # Correct reward is the last value of r.
        
        end = datetime.now()
        time_diff = (end - start).total_seconds()
        total_time += time_diff
        logger.info(f'Took {time_diff} seconds for one compression.')

    logger.info(f'Evaluation of {n_games} took {total_time} secs. An average of {total_time/n_games} secs per game.')

    return np.mean(rewards), np.mean(acc), np.mean(weights)


num_datasets = len(dataset_names)

num_tests = (rl_iterations//10) + 1

weights_history_tests = np.zeros(shape=(num_tests, num_datasets))
acc_history_tests = np.zeros(shape=(num_tests, num_datasets))
rw_history_tests = np.zeros(shape=(num_tests, num_datasets))
test_counter = 1

conv_exp_replay = PrioritizedExperienceReplayBufferMultipleDatasets(replay_buffer_size, dataset_names)


for idx, env in enumerate(envs):
    weights_history_tests[0, idx ] = env.weights_before
    acc_history_tests[0, idx] = env.test_acc_before


highest_td_error = True
highest_rw = 0

with tqdm(total=rl_iterations,
        bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Last 3 RW: {postfix[0][0]:.2f}, {postfix[0][1]:.2f} & {postfix[0][2]:.2f} W: {postfix[1][0]:.2f}, {postfix[1][1]:.2f} & {postfix[1][2]:.2f} Acc: {postfix[2][0]:.2f}, {postfix[2][1]:.2f} & {postfix[2][2]:.2f}] Replay: conv:{postfix[3]}. Epsylon={postfix[4]}.",
        postfix=[dict({0: 0, 1: 0, 2: np.mean(rw_history_tests[0])}), 
        dict({0: 0, 1: 0, 2: np.mean(weights_history_tests[0])}),
        dict({0: 0, 1: 0, 2: np.mean(acc_history_tests[0])}), len(conv_exp_replay), conv_agent.epsilon]) as t:

    for i in range(rl_iterations):

        # generate new sample
        for idx, env in enumerate(envs):
            dataset = dataset_names[idx]
            play_and_record(conv_agent, env, conv_exp_replay, run_id=run_id, test_number=i, dataset_name=dataset,save_name=exploration_filename,n_games=1)

        for w_it in range(update_weights_iterations):
            logger.debug(f'Processing batch {w_it+1}/{update_weights_iterations}.')
            
            
            logger.debug('Training conv agent.')
            batch_data = sample_batch(conv_exp_replay, batch_size=rl_batch_size, highest_td_error=highest_td_error)
            td_error_indexes = batch_data['td_indexes']
            del batch_data['td_indexes']
            conv_loss_t = update_agent_conv(**batch_data)
            conv_exp_replay.update_td_error(td_error_indexes, np.abs(conv_loss_t.numpy().flatten()))
           


        td_loss_history_conv.append(np.mean(conv_loss_t))

        # highest_td_error = not highest_td_error

        conv_agent.epsilon = max(min_epsylon, conv_agent.epsilon * epsylon_decay )

        t.postfix[3] = len(conv_exp_replay)
        t.postfix[4] = conv_agent.epsilon
        
        # adjust agent parameters
        if i % test_frequency_epochs == 0:
            accumulated_rw = 0
            for idx, env in enumerate(test_envs):
                dataset = dataset_names[idx]
                logger.debug(f'Testing for dataset {dataset}.')
                rw, acc, weights = play_and_record(conv_agent, env, conv_exp_replay, run_id=run_id,test_number=i//10, dataset_name=dataset,save_name=test_filename, n_games=eval_n_samples, exploration=False)            
                rw_history_tests[test_counter, idx] = rw
                acc_history_tests[test_counter, idx] = acc
                accumulated_rw += rw
                weights_history_tests[test_counter, idx] = weights
                
            accumulated_rw = accumulated_rw / num_datasets
            
            conv_target_network.model.set_weights(conv_agent.model.get_weights())
            if accumulated_rw > highest_rw:
                highest_rw = accumulated_rw
   
                conv_agent.model.save_weights(agents_path+'_conv.cpkt')
                conv_target_network.model.save_weights(agents_path+'_conv_target.ckpt')


            t.postfix[0][2] = np.mean(rw_history_tests[test_counter])
            try:
                t.postfix[0][1] = np.mean(rw_history_tests[test_counter-1])
            except IndexError:
                t.postfix[0][1] = 0
            try:
                t.postfix[0][0] = np.mean(rw_history_tests[test_counter-2])
            except IndexError:
                t.postfix[0][0] = 0

            t.postfix[1][2] = np.mean(weights_history_tests[test_counter])
            try:
                t.postfix[1][1] = np.mean(weights_history_tests[test_counter-1])
            except IndexError:
                t.postfix[2][1] = 0
            try:
                t.postfix[1][0] = np.mean(weights_history_tests[test_counter-2])
            except IndexError:
                t.postfix[2][0] = 0

            t.postfix[2][2] = np.mean(acc_history_tests[test_counter])
            try:
                t.postfix[2][1] = np.mean(acc_history_tests[test_counter-1])
            except IndexError:
                t.postfix[2][1] = 0
            try:
                t.postfix[2][0] = np.mean(acc_history_tests[test_counter-2])
            except IndexError:
                t.postfix[2][0] = 0

            test_counter += 1
            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            ax1.title.set_text('Accuracy')
            for idx, dataset_name in enumerate(dataset_names):
                ax1.plot(acc_history_tests[:test_counter, idx])
            ax1.legend(dataset_names)
            ax2.title.set_text('Weights')
            for idx, dataset_name in enumerate(dataset_names):
                ax2.plot(weights_history_tests[:test_counter, idx])
            ax2.legend(dataset_names)
            ax3.title.set_text('Reward')
            for idx, dataset_name in enumerate(dataset_names):
                ax3.plot(rw_history_tests[:test_counter, idx])
            ax3.legend(dataset_names)
            plt.xlabel('Epochs')
            plt.savefig(figures_path+'_test_stats.png')
            plt.close()

        t.update()
        

        plt.plot(td_loss_history_conv, color='r')
        plt.legend(['conv'])
        plt.ylabel("TD loss")
        plt.xlabel('Epochs')
        plt.savefig(figures_path+'_td_loss.png')
        plt.close()