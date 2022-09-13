import tensorflow as tf
import logging

from CompressionLibrary.agent_evaluators import make_env_imagenet, evaluate_agents, play_and_record
from CompressionLibrary.reinforcement_models import RandomAgent
from CompressionLibrary.utils import load_dataset

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

eval_n_samples = 10
n_samples_mode = 128
batch_size_per_replica = 128
tuning_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync


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

random_conv = RandomAgent('random_conv', conv_n_actions)
random_fc = RandomAgent('random_fc', fc_n_actions)

results = evaluate_agents(env, random_conv,random_fc, 'test_env', 1, save_name=data_path+'/test_evaluate.csv', n_games=5)

print(results)
