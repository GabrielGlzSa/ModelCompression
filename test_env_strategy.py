import tensorflow as tf
import logging
import tensorflow_datasets as tfds

from CompressionLibrary.agent_evaluators import make_env_imagenet, evaluate_agents, play_and_record
from CompressionLibrary.environments import *
from CompressionLibrary.reinforcement_models import RandomAgent



data_path = '/mnt/disks/mcdata/data'
# data_path = './data'

# Use below for TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
strategy = tf.distribute.TPUStrategy(resolver)
# Use below for GPU
# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

dataset = 'imagenet2012'
current_state = 'layer_input'
next_state = 'layer_output'

eval_n_samples = 10
n_samples_mode = 64
batch_size_per_replica = 64
tuning_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync


layer_name_list = [ 'block2_conv1', 'block2_conv2', 
                    'block3_conv1', 'block3_conv2', 'block3_conv3',
                    'block4_conv1', 'block4_conv2', 'block4_conv3',
                    'block5_conv1', 'block5_conv2', 'block5_conv3',
                    'fc1', 'fc2']



def resize_image(image, shape = (224,224)):
  target_width = shape[0]
  target_height = shape[1]
  initial_width = tf.shape(image)[0]
  initial_height = tf.shape(image)[1]
  im = image
  ratio = 0
  if(initial_width < initial_height):
    ratio = tf.cast(256 / initial_width, tf.float32)
    h = tf.cast(initial_height, tf.float32) * ratio
    im = tf.image.resize(im, (256, h), method="bicubic")
  else:
    ratio = tf.cast(256 / initial_height, tf.float32)
    w = tf.cast(initial_width, tf.float32) * ratio
    im = tf.image.resize(im, (w, 256), method="bicubic")
  width = tf.shape(im)[0]
  height = tf.shape(im)[1]
  startx = width//2 - (target_width//2)
  starty = height//2 - (target_height//2)
  im = tf.image.crop_to_bounding_box(im, startx, starty, target_width, target_height)
  return im

@tf.function
def imagenet_preprocessing(img, label):
    img = tf.cast(img, tf.float32)
    img = resize_image(img)
    img = tf.keras.applications.vgg16.preprocess_input(img, data_format=None)
    return img, label

splits, info = tfds.load('imagenet2012', as_supervised=True, with_info=True, shuffle_files=True, 
                            split=['train[:80%]', 'train[80%:]', 'test'], data_dir=data_path)

(train_examples, validation_examples, test_examples) = splits
num_examples = info.splits['train'].num_examples

num_classes = info.features['label'].num_classes
input_shape = info.features['image'].shape


def create_model():
    optimizer = tf.keras.optimizers.Adam(1e-5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()


    model = tf.keras.applications.vgg16.VGG16(
                            include_top=True,
                            weights='imagenet',
                            input_shape=(224,224,3),
                            classes=num_classes,
                            classifier_activation='softmax'
                        )
    model.compile(optimizer=optimizer, loss=loss_object,
                    metrics=train_metric)

    model.summary()     

    return model       

train_ds = train_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=2048, reshuffle_each_iteration=True).batch(tuning_batch_size).prefetch(tf.data.AUTOTUNE)
valid_ds = validation_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(tuning_batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(tuning_batch_size).prefetch(tf.data.AUTOTUNE)


input_shape = (224,224,3)
env = make_env_imagenet(create_model, train_ds, valid_ds, test_ds, input_shape, layer_name_list, n_samples_mode, tuning_batch_size, current_state, next_state, strategy)
env.model.summary()

conv_shape, dense_shape = env.observation_space()
conv_n_actions, fc_n_actions = env.action_space()

print(conv_shape, dense_shape)

random_agent = RandomAgent('random', max(conv_n_actions, fc_n_actions))

results = evaluate_agents(env, random_agent,random_agent, save_name='./data/test_evaluate.csv')

print(results)
