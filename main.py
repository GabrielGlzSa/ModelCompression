import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import logging
from CompressionTechniques import *

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.get_logger().setLevel('ERROR')

# logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logger = logging.getLogger(__name__)

tfds.disable_progress_bar()

splits, info = tfds.load('cifar10', as_supervised=True, with_info=True,
                         split=['train[:80%]', 'train[80%:]', 'test'], data_dir='./data')

(train_examples, validation_examples, test_examples) = splits

print(info)
num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
input_shape = info.features['image'].shape
BATCH_SIZE = 32
IMAGE_SIZE = None
input_shape = list(input_shape)
print(input_shape)
if input_shape[0]>224:
    IMAGE_SIZE = 224
else:
    IMAGE_SIZE = input_shape[0]

input_shape[0] = IMAGE_SIZE
input_shape[1] = IMAGE_SIZE

print('Number of examples', num_examples)
print('Number of classes', num_classes)


@tf.function
def map_fn(img, label):
    img = tf.image.resize(img, size=(IMAGE_SIZE, IMAGE_SIZE))
    img /= 255.

    return img, label


def prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn, batch_size):
    train_ds = train_examples.shuffle(buffer_size=num_examples).map(map_fn).batch(batch_size)
    valid_ds = validation_examples.map(map_fn).batch(batch_size)
    test_ds = test_examples.map(map_fn).batch(batch_size)

    return train_ds, valid_ds, test_ds


train_ds, valid_ds, test_ds = prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn,
                                              BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

epochs = 1


def create_model():
    return tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_0'),
                                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
                                tf.keras.layers.MaxPool2D((2, 2), 2),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(128, activation='relu', name='dense_0'),
                                tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
                                tf.keras.layers.Dense(num_classes, activation='softmax', name='softmax')])


model = create_model()
model.compile(optimizer=optimizer, loss=loss_object, metrics=train_metric)
history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds)  # _generator

# https://www.cs.utexas.edu/users/risto/

# Pruned 127 weights and kept same accuracy 1.0.
# compressor = DeepCompression(model=model, threshold=0.001)
# compressor.compress_layer('Dense2')

# compressor = ReplaceDenseWithGlobalAvgPool(model=model, dataset=train_ds,
#                                            optimizer=optimizer, loss=loss_object, metrics=train_metric)
# compressor.compress_layer()


# compressor = InsertDenseSVD(model=model, dataset=train_ds,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric, fine_tune=False)
# compressor.compress_layer('dense_1', units=32)


# compressor = InsertDenseSparse(model=model, dataset=train_ds,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric)
# compressor.compress_layer('dense_1', verbose=True)


# compressor = InsertSVDConv(model=model, dataset=train_ds,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric)
# compressor.compress_layer('conv2d_1')

# compressor = DepthwiseSeparableConvolution(model=model, dataset=train_ds,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric)
# compressor.compress_layer('conv2d_1')

# compressor = FireLayerCompression(model=model, dataset=train_ds,
#                                   optimizer=optimizer, loss=loss_object, metrics=train_metric)
# compressor.compress_layer('conv2d_1')

# compressor = MLPCompression(model=model, dataset=train_ds,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric)
# compressor.compress_layer('conv2d_1')

# population=20,generations=100 takes 3 hours and 40 minutes
# compressor = SparseConnectionsCompressionCustom(model=model,
#                                                 dataset=train_ds,
#                                                 optimizer=optimizer,
#                                                 loss=loss_object,
#                                                 metrics=train_metric)
# compressor.compress_layer('conv2d_1', epochs=10)

# RAM OOM in training loop.
compressor = SparseConvolutionCompression(model=model, dataset=train_ds,
                            optimizer=optimizer, loss=loss_object, metrics=train_metric, bases=3)
compressor.compress_layer('conv2d_1', iterations=2)

new_model = compressor.get_model()
for layer in new_model.layers:
    print(layer)
