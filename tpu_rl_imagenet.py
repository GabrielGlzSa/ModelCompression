import tensorflow as tf
import logging

from CompressionLibrary.agent_evaluators import make_env_imagenet, evaluate_agents, play_and_record
from CompressionLibrary.reinforcement_models import DQNAgent
from CompressionLibrary.replay_buffer import ReplayBuffer
from CompressionLibrary.utils import load_dataset

from uuid import uuid4
from datetime import datetime
from tqdm import tqdm
import pandas as pd


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
  

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

dataset = 'imagenet2012'
current_state = 'layer_input'
next_state = 'layer_output'

epsilon_start_value = 0.9
epsilon_decay = 0.999
min_epsilon = 0.1
replay_buffer_size = 10 ** 5

rl_iterations = 10000
eval_n_samples = 10
n_samples_mode = 128
batch_size_per_replica = 128
tuning_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
rl_batch_size = tuning_batch_size

layer_name_list = [ 'block2_conv1', 'block2_conv2', 
                    'block3_conv1', 'block3_conv2', 'block3_conv3',
                    'block4_conv1', 'block4_conv2', 'block4_conv3',
                    'block5_conv1', 'block5_conv2', 'block5_conv3',
                    'fc1', 'fc2']


def create_model():
    optimizer = tf.keras.optimizers.Adam(1e-5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()


    model = tf.keras.applications.vgg16.VGG16(
                            include_top=True,
                            weights='imagenet',
                            input_shape=(224,224,3),
                            classes=1000,
                            classifier_activation='softmax'
                        )
    model.compile(optimizer=optimizer, loss=loss_object,
                    metrics=train_metric)

    return model       

train_ds, valid_ds, test_ds, input_shape, _ = load_dataset(data_path)


input_shape = (224,224,3)
env = make_env_imagenet(create_model, train_ds, valid_ds, test_ds, input_shape, layer_name_list, n_samples_mode, tuning_batch_size, current_state, next_state, strategy, data_path)
env.model.summary()

conv_shape, dense_shape = env.observation_space()
conv_n_actions, fc_n_actions = env.action_space()

print(conv_shape, dense_shape)





def load_weigths_into_target_network(agent, target_network):
    """ assign target_network.weights variables to their respective agent.weights values. """
    for i in range(len(agent.model.layers)):
        target_network.model.layers[i].set_weights(
            agent.model.layers[i].get_weights())





def moving_average(x, span=100, **kw):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span, **kw).mean().values


mean_weights_history = []
mean_acc_history = []
mean_rw_history = []
td_loss_history_conv = []
td_loss_history_fc = []


def sample_batch(exp_replay, batch_size):
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(
        batch_size)
    return {
        'state': obs_batch,
        'action': act_batch,
        'rewards': reward_batch,
        'next_state': next_obs_batch,
        'done': is_done_batch,
    }


@tf.function
def training_loop(state, action, rewards, next_state, done, agent, target_agent, loss_optimizer, n_actions, gamma=0.01):
    state = tf.cast(state, tf.float32)
    action = tf.cast(action, tf.int32)
    next_state = tf.cast(next_state, tf.float32)

    rewards = tf.cast(rewards, tf.float32)
    done = 1 - tf.cast(done, tf.float32)

    reference_qvalues = rewards + gamma * \
        tf.reduce_max(target_agent.get_qvalues(next_state), axis=1)
    reference_qvalues = reference_qvalues * (1 - done) - done

    masks = tf.one_hot(action, n_actions)
    with tf.GradientTape() as tape:
        q_values = agent.get_qvalues(state)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        td_loss = tf.reduce_mean((q_action - reference_qvalues) ** 2)

    gradients = tape.gradient(td_loss, agent.weights)
    loss_optimizer.apply_gradients(zip(gradients, agent.weights))
    return td_loss

fc_exp_replay = ReplayBuffer(replay_buffer_size)
conv_exp_replay = ReplayBuffer(replay_buffer_size)

with strategy.scope():
    
    fc_agent = DQNAgent("dqn_agent_fc", dense_shape,
                        fc_n_actions, epsilon=epsilon_start_value, layer_type='fc')
    fc_target_network = DQNAgent(
        "target_network_fc", dense_shape, fc_n_actions, layer_type='fc')


    conv_agent = DQNAgent("dqn_agent_conv", conv_shape,
                        conv_n_actions, epsilon=epsilon_start_value, layer_type='cnn')
    conv_target_network = DQNAgent(
        "target_network_conv", conv_shape, conv_n_actions, layer_type='cnn')


    try:
        conv_target_network.model.load_weights(
            './data/checkpoints/{}_my_checkpoint_conv'.format(dataset))
        fc_target_network.model.load_weights(
            './data/checkpoints/{}_my_checkpoint_fc'.format(dataset))
    except:
        print('Failed to find pretrained models for the RL agents.')
        pass

    load_weigths_into_target_network(fc_agent, fc_target_network)
    load_weigths_into_target_network(conv_agent, conv_target_network)

    for w, w2 in zip(fc_agent.model.trainable_weights, fc_target_network.model.trainable_weights):
        tf.assert_equal(w, w2)
    for w, w2 in zip(conv_agent.model.trainable_weights, conv_target_network.model.trainable_weights):
        tf.assert_equal(w, w2)

    print("It works!")

    optimizer = tf.keras.optimizers.Adam(1e-5)


    print('There are {} conv and {} fc instances.'.format(len(conv_exp_replay), len(fc_exp_replay)))
    play_and_record(conv_agent, fc_agent, env, conv_exp_replay, fc_exp_replay, n_steps=1)

    print('There are {} conv and {} fc instances.'.format(len(conv_exp_replay), len(fc_exp_replay)))

    with tqdm(total=rl_iterations,
            bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Epsilon: {postfix[0]:.4f} Last 3 RW: {postfix[1][0]:.2f}, {postfix[1][1]:.2f} & {postfix[1][2]:.2f} W: {postfix[2][0]:.2f}, {postfix[2][1]:.2f} & {postfix[2][2]:.2f} Acc: {postfix[3][0]:.2f}, {postfix[3][1]:.2f} & {postfix[3][2]:.2f}]",
            postfix=[conv_agent.epsilon, dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0})]) as t:
        for i in range(rl_iterations):
            # generate new sample
            play_and_record(conv_agent, fc_agent, env, conv_exp_replay, fc_exp_replay, n_steps=1)

            # train fc
            batch_data = sample_batch(conv_exp_replay, batch_size=rl_batch_size)
            batch_data['agent'] = conv_agent
            batch_data['target_agent'] = conv_target_network
            batch_data['loss_optimizer'] = optimizer
            batch_data['n_actions'] = conv_n_actions
            conv_loss_t = training_loop(**batch_data)
            td_loss_history_conv.append(conv_loss_t)

            # train
            batch_data = sample_batch(fc_exp_replay, batch_size=rl_batch_size)
            batch_data['agent'] = fc_agent
            batch_data['target_agent'] = fc_target_network
            batch_data['loss_optimizer'] = optimizer
            batch_data['n_actions'] = fc_n_actions
            fc_loss_t = training_loop(**batch_data)
            td_loss_history_fc.append(fc_loss_t)

            # adjust agent parameters
            if i % 10 == 0:
                load_weigths_into_target_network(conv_agent, conv_target_network)
                conv_target_network.model.save_weights(
                    data_path+'/checkpoints/{}_my_checkpoint_conv_agent'.format(dataset))
                conv_agent.epsilon = max(conv_agent.epsilon * epsilon_decay, min_epsilon)
                t.postfix[0] = conv_agent.epsilon

                load_weigths_into_target_network(fc_agent, fc_target_network)
                fc_target_network.model.save_weights(
                     data_path+'/checkpoints/{}_my_checkpoint_fc_agent'.format(dataset))
                fc_agent.epsilon = max(fc_agent.epsilon * epsilon_decay, min_epsilon)

                rw, acc, weights = evaluate_agents(env, fc_agent, conv_agent, run_id=run_id,test_number=i/10, save_name=data_path+'/test_evaluate.csv', n_games=4)            
                mean_rw_history.append(rw)
                mean_acc_history.append(acc)
                mean_weights_history.append(weights)

                t.postfix[1][2] = mean_rw_history[-1]
                try:
                    t.postfix[1][1] = mean_rw_history[-2]
                except IndexError:
                    t.postfix[1][1] = 0
                try:
                    t.postfix[1][0] = mean_rw_history[-3]
                except IndexError:
                    t.postfix[1][0] = 0

                t.postfix[2][2] = mean_weights_history[-1]
                try:
                    t.postfix[2][1] = mean_weights_history[-2]
                except IndexError:
                    t.postfix[2][1] = 0
                try:
                    t.postfix[2][0] = mean_weights_history[-3]
                except IndexError:
                    t.postfix[2][0] = 0

                t.postfix[3][2] = mean_acc_history[-1]
                try:
                    t.postfix[3][1] = mean_acc_history[-2]
                except IndexError:
                    t.postfix[3][1] = 0
                try:
                    t.postfix[3][0] = mean_acc_history[-3]
                except IndexError:
                    t.postfix[3][0] = 0
            t.update()