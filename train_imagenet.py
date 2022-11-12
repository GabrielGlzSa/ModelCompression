
import tensorflow as tf
import logging

from CompressionLibrary.agent_evaluators import evaluate_agents, play_and_record
from CompressionLibrary.reinforcement_models import DDPG
from CompressionLibrary.replay_buffer import ReplayBuffer
from CompressionLibrary.utils import load_dataset
from CompressionLibrary.environments import ModelCompressionSVDEnv

from uuid import uuid4
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt



run_id = datetime.now().strftime('%Y-%m-%d-%H-%M%S-') + str(uuid4())

dataset_name = 'imagenet2012_subset'

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


logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(f'/home/A00806415/DCC/ModelCompression/data/ModelCompression_{dataset_name}.log', 'w+')],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)



exploration_filename = data_path+'/training_exploration_DDPG.csv'
test_filename = data_path+'/test_evaluate_DDPG.csv'
agents_path = data_path+'/checkpoints/{}_my_checkpoint_DDPG_'.format(dataset_name)

current_state = 'layer_input'
next_state = 'layer_output'

epsilon_start_value = 0.9
epsilon_decay = 0.995
min_epsilon = 0.1
replay_buffer_size = 10 ** 5

rl_iterations = 1000
eval_n_samples = 5
n_samples_mode = 128
batch_size_per_replica = 128
tuning_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
rl_batch_size = tuning_batch_size
verbose = 1
tuning_mode = 'layer'

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

train_ds, valid_ds, test_ds, input_shape, _, num_examples = load_dataset(data_path, 32)

print(f'Using {n_samples_mode} examples.')

input_shape = (224,224,3)

def make_env(create_model, train_ds, valid_ds, test_ds, input_shape, layer_name_list, num_feature_maps, tuning_batch_size, tuning_epochs, verbose=0, tuning_mode='layer', current_state_source='layer_input', next_state_source='layer_output', strategy=None, model_path='./data'):

    w_comprs = ['InsertDenseSVD'] 
    l_comprs = ['MLPCompression']
    compressors_list = w_comprs +  l_comprs

    parameters = {}
    parameters['InsertDenseSVD'] = {'layer_name': None}
    parameters['MLPCompression'] = {'layer_name': None}

    env = ModelCompressionSVDEnv(compressors_list, create_model, parameters,
                                      train_ds, valid_ds, test_ds,
                                      layer_name_list, input_shape, current_state_source=current_state_source, next_state_source=next_state_source, 
                                      num_feature_maps=num_feature_maps, tuning_batch_size=tuning_batch_size, verbose=verbose, tuning_mode=tuning_mode, tuning_epochs=tuning_epochs,strategy=strategy, model_path=model_path)

    return env

env = make_env(
        create_model=create_model, 
        train_ds=train_ds, 
        valid_ds=valid_ds, 
        test_ds=test_ds, 
        input_shape=input_shape, 
        layer_name_list=layer_name_list, 
        num_feature_maps=n_samples_mode, 
        tuning_batch_size=tuning_batch_size, 
        tuning_epochs=0, 
        verbose=verbose, 
        tuning_mode='layer', 
        current_state_source=current_state, 
        next_state_source=next_state, 
        strategy=strategy, 
        model_path=data_path)

env.model.summary()

conv_shape, dense_shape = env.observation_space()
conv_n_actions, fc_n_actions = env.action_space()

print(conv_shape, dense_shape)

def load_weigths_into_target_network(agent, target_network):
    """ assign target_network.weights variables to their respective agent.weights values. """
    for i in range(len(agent.model.layers)):
        target_network.model.layers[i].set_weights(
            agent.model.layers[i].get_weights())




with strategy.scope():
    
    fc_agent = DDPG("ddpg_agent_fc", dense_shape,
                        fc_n_actions, epsilon=epsilon_start_value, layer_type='fc')
    fc_target_network = DDPG(
        "target_network_fc", dense_shape, fc_n_actions, layer_type='fc')

    fc_agent.model.summary()

    conv_agent = DDPG("ddpg_agent_conv", conv_shape,
                        conv_n_actions, epsilon=epsilon_start_value, layer_type='cnn')
    conv_target_network = DDPG(
        "target_network_conv", conv_shape, conv_n_actions, layer_type='cnn')


    try:
        conv_target_network.model.load_weights(
            agents_path+'conv_agent')
        fc_target_network.model.load_weights(
            agents_path+'fc_agent'.format(dataset_name))
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

    optimizer_conv = tf.keras.optimizers.Adam()
    optimizer_fc = tf.keras.optimizers.Adam()

@tf.function
def training_loop(state, action, rewards, next_state, agent, target_agent, actor_optimizer, critic_optimizer, gamma=0.01):
    with tf.GradientTape() as tape:
        target_actions = agent.actor(next_state, training=True)
        y = rewards + gamma * target_agent.critic([next_state, target_actions], training=True)
        critic_value = agent.critic([state, action], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    gradients = tape.gradient(critic_loss, agent.critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, agent.critic.trainable_variables))

    with tf.GradientTape() as tape:
        actions = agent.actor(state, training=True)
        critic_value = agent.critic([state, actions], training = True)

        actor_loss = -tf.math.reduce_mean(critic_value)

    gradients = tape.gradient(actor_loss, agent.actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, agent.actor.trainable_variables))

    return critic_loss, actor_loss

mean_weights_history = []
mean_acc_history = []
mean_rw_history = []
td_loss_history_conv = []
td_loss_history_fc = []

def sample_batch(exp_replay, batch_size):
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(
        batch_size)
    return {
        'state': tf.cast(obs_batch.to_tensor(), tf.float32),
        'action': tf.cast(act_batch, tf.int32),
        'rewards': tf.cast(reward_batch, tf.float32),
        'next_state': tf.cast(next_obs_batch.to_tensor(), tf.float32),
        'done': tf.cast(is_done_batch, tf.float32),
    }
    

fc_exp_replay = ReplayBuffer(replay_buffer_size)
conv_exp_replay = ReplayBuffer(replay_buffer_size)

print('There are {} conv and {} fc instances.'.format(len(conv_exp_replay), len(fc_exp_replay)))
play_and_record(conv_agent, fc_agent, env, conv_exp_replay, fc_exp_replay, run_id=run_id, test_number=0, dataset_name=dataset_name,save_name=exploration_filename, n_games=5)

print('There are {} conv and {} fc instances.'.format(len(conv_exp_replay), len(fc_exp_replay)))

with tqdm(total=rl_iterations,
        bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Epsilon: {postfix[0]:.4f} Last 3 RW: {postfix[1][0]:.2f}, {postfix[1][1]:.2f} & {postfix[1][2]:.2f} W: {postfix[2][0]:.2f}, {postfix[2][1]:.2f} & {postfix[2][2]:.2f} Acc: {postfix[3][0]:.2f}, {postfix[3][1]:.2f} & {postfix[3][2]:.2f}] Replay: conv:{postfix[4]}/fc:{postfix[5]}.",
        postfix=[conv_agent.epsilon, dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0}), dict({0: 0, 1: 0, 2: 0}), len(conv_exp_replay), len(fc_exp_replay)]) as t:
    for i in range(rl_iterations):
        # generate new sample
        play_and_record(conv_agent, fc_agent, env, conv_exp_replay, fc_exp_replay, run_id=run_id, test_number=i, dataset_name=dataset_name,save_name=exploration_filename,n_games=1)


        # train fc
        batch_data = sample_batch(conv_exp_replay, batch_size=rl_batch_size)
        batch_data['agent'] = conv_agent
        batch_data['target_agent'] = conv_target_network
        batch_data['loss_optimizer'] = optimizer_conv
        batch_data['n_actions'] = conv_n_actions
        conv_loss_t = training_loop(**batch_data)
        td_loss_history_conv.append(conv_loss_t)

        # train
        batch_data = sample_batch(fc_exp_replay, batch_size=rl_batch_size)
        batch_data['agent'] = fc_agent
        batch_data['target_agent'] = fc_target_network
        batch_data['loss_optimizer'] = optimizer_fc
        batch_data['n_actions'] = fc_n_actions
        fc_loss_t = training_loop(**batch_data)
        td_loss_history_fc.append(fc_loss_t)


        conv_agent.epsilon = max(conv_agent.epsilon * epsilon_decay, min_epsilon)
        fc_agent.epsilon = max(fc_agent.epsilon * epsilon_decay, min_epsilon)
        t.postfix[0] = conv_agent.epsilon
        t.postfix[5]= len(fc_exp_replay)
        t.postfix[4]= len(conv_exp_replay)
        # adjust agent parameters
        if i % 10 == 0:
            load_weigths_into_target_network(conv_agent, conv_target_network)
            conv_target_network.model.save_weights(
                data_path+'/checkpoints/{}_my_checkpoint_conv_agent'.format(dataset_name))
            
            
            load_weigths_into_target_network(fc_agent, fc_target_network)
            fc_target_network.model.save_weights(
                    data_path+'/checkpoints/{}_my_checkpoint_fc_agent'.format(dataset_name))
            
            

            rw, acc, weights = evaluate_agents(env, conv_agent, fc_agent,run_id=run_id,test_number=i//10, dataset_name=dataset_name,save_name=test_filename, n_games=eval_n_samples)            
            
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



            plt.clf()
            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            ax1.title.set_text('Accuracy')
            ax1.plot(mean_acc_history)
            ax2.title.set_text('Weights(%)')
            ax2.plot(mean_weights_history)
            ax3.title.set_text('Reward')
            ax3.plot(mean_rw_history)
            plt.savefig('./data/figures/test_stats.png')

        t.update()
        
        plt.clf()
        plt.plot(td_loss_history_conv, color='r')
        plt.plot(td_loss_history_fc, color='b')
        plt.savefig('./data/figures/td_loss.png')
   