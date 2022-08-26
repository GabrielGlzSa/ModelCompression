from unittest import result
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
from CompressionLibrary.utils import calculate_model_weights
from CompressionLibrary.CompressionTechniques import *
from CompressionLibrary.custom_callbacks import EarlyStoppingReward



resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
strategy = tf.distribute.TPUStrategy(resolver)

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
                            split=['train[:80%]', 'train[80%:]', 'test'], data_dir='./data')

(train_examples, validation_examples, test_examples) = splits
num_examples = info.splits['train'].num_examples

num_classes = info.features['label'].num_classes
input_shape = info.features['image'].shape

input_shape = (224,224,3)
batch_size = 256

train_ds = train_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
valid_ds = validation_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)



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

with strategy.scope():
  model = create_model()
  loss, acc_before = model.evaluate(valid_ds)


weights_before = calculate_model_weights(model)



reward_before = acc_before

callbacks = []

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# tf.get_logger().setLevel('ERROR')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

tfds.disable_progress_bar()

new_layers = []
optimizer = tf.keras.optimizers.Adam(1e-5)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Pruned 127 weights and kept same accuracy 1.0.
# compressor = DeepCompression(model=model, dataset=train_ds,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
# compressor.compress_layer('fc1', threshold=0.001)

# compressor = ReplaceDenseWithGlobalAvgPool(model=model, dataset=train_ds, 
#                                            optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
# compressor.compress_layer('fc1')
# model = compressor.get_model()
# new_layers.append(compressor.new_layer_name)


compressor = InsertSVDConv(model=model, dataset=train_ds,
                            optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
compressor.compress_layer('block2_conv1')
model = compressor.get_model()
new_layers.append(compressor.new_layer_name)


compressor = DepthwiseSeparableConvolution(model=model, dataset=train_ds, tuning_epochs=10,
                            optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
compressor.compress_layer('block2_conv2')
model = compressor.get_model()
new_layers.append(compressor.new_layer_name)



compressor = FireLayerCompression(model=model, dataset=train_ds, tuning_epochs=10,
                                  optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
compressor.compress_layer('block3_conv1')
model = compressor.get_model()
new_layers.append(compressor.new_layer_name)



compressor = MLPCompression(model=model, dataset=train_ds, tuning_epochs=10,
                            optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)

compressor.compress_layer('block3_conv2')
model = compressor.get_model()
new_layers.append(compressor.new_layer_name)



compressor = SparseConnectionsCompression(model=model, dataset=train_ds, tuning_epochs=10, num_batches=100, tuning_verbose=1, fine_tuning=False,
                                          optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape, callbacks=callbacks)
compressor.compress_layer('block5_conv1',target_perc=0.75, conn_perc_per_epoch=0.15)
model = compressor.get_model()
new_layers.append(compressor.new_layer_name)
callbacks = compressor.callbacks

compressor = SparseConnectionsCompression(model=model, dataset=train_ds, tuning_epochs=10, num_batches=100, tuning_verbose=1, fine_tuning=False,
                                          optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape, callbacks=callbacks)
compressor.compress_layer('block5_conv3',target_perc=0.75, conn_perc_per_epoch=0.15)
new_layers.append(compressor.new_layer_name)
model = compressor.get_model()
callbacks = compressor.callbacks

# compressor = SparseConvolutionCompression(model=model, dataset=train_ds, tuning_epochs=10, fine_tuning=True, num_batches=100,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
# compressor.compress_layer('block2_conv1', new_layer_iterations=10000, new_layer_iterations_sparse=30000, new_layer_verbose=True)



compressor = InsertDenseSVD(model=model, dataset=train_ds, tuning_epochs=4,
                            optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
compressor.compress_layer('fc1')
model = compressor.get_model()
new_layers.append(compressor.new_layer_name)



# compressor = InsertDenseSparse(model=model, dataset=train_ds,  tuning_epochs=10,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
# compressor.compress_layer('fc2', new_layer_iterations=20000, new_layer_verbose=True)
# model = compressor.get_model()
# new_layers.append(compressor.new_layer_name)
# model.summary()


# compressor = SparseConvolutionCompression(model=model, dataset=train_ds, tuning_epochs=1, num_batches=10,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
# compressor.compress_layer('block3_conv1')


for layer in model.layers:
  if layer.name in new_layers:
    layer.trainable=True
  else:
    layer.trainable=False

model.summary()

Rcb = EarlyStoppingReward(acc_before=acc_before, weights_before=weights_before, verbose=1)

callbacks.append(Rcb)

model_path = '/mnt/mcdata/data/test_tpu_save'
model.save(model_path)

with strategy.scope():
  model = tf.keras.models.load_model(model_path)
  model.fit(train_ds, epochs=20, validation_data=valid_ds, callbacks=callbacks)
  loss, valid_acc = model.evaluate(valid_ds)
  weights_after = calculate_model_weights(model)
  valid_reward = 1 - (weights_after/weights_before) + valid_acc

  print(f'Validation reward is {valid_reward}. The model has {weights_after} weights and {valid_acc} accuracy.')
  loss, test_acc = model.evaluate(test_ds)
  test_reward = 1 - (weights_after/weights_before) + test_acc
  print(f'Validation reward is {test_reward}. The model has {weights_after} weights and {test_acc} accuracy.')