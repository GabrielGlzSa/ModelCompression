import os
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import tensorflow_datasets as tfds
import logging
import matplotlib.pyplot as plt
from CompressionTechniques import *

#logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logger = logging.getLogger(__name__)

tfds.disable_progress_bar()

splits, info = tfds.load('horses_or_humans', as_supervised=True, with_info=True,
                         split=['train[:80%]', 'train[80%:]', 'test'], data_dir='./data')

(train_examples, validation_examples, test_examples) = splits

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

BATCH_SIZE = 32
IMAGE_SIZE = 224

print('Number of examples', num_examples)
print('Number of classes', num_classes)


@tf.function
def map_fn(img, label):
    image_height = 224
    image_width = 224
    img = tf.image.resize(img, size=(image_height, image_width))
    img /= 255.

    return img, label


def prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn, batch_size):
    train_ds = train_examples.shuffle(buffer_size=num_examples).map(map_fn).batch(batch_size)
    valid_ds = validation_examples.map(map_fn).batch(batch_size)
    test_ds = test_examples.map(map_fn).batch(batch_size)

    return train_ds, valid_ds, test_ds


train_ds, valid_ds, test_ds = prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn,
                                              BATCH_SIZE)


def test_model(layers, dataset, loss_object, metric_object):
  metric_object.reset_state()
  losses = []
  for x,y in dataset:
    for layer in layers:
      x = layer(x)
    losses.append(loss_object(y, x))
    metric_object.update_state(y,x)
  loss = tf.reduce_sum(losses)
  metric = metric_object.result()
  return loss, metric


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

epochs = 2
def create_model():
  return tf.keras.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu', name='conv2d_0'),
                         tf.keras.layers.Conv2D(16,(3,3),activation='relu', name='conv2d_1'),
                         tf.keras.layers.MaxPool2D((2,2), 2),
                         tf.keras.layers.Flatten(),
                         tf.keras.layers.Dense(32,activation='relu', name='Dense1'),
                         tf.keras.layers.Dense(32,activation='relu', name='Dense2'),
                         tf.keras.layers.Dense(num_classes, activation='softmax', name='DenseSoftmax')
  ])

# def create_MLP():
#   return tf.keras.Sequential([MLPConv(filters=32, kernel_size=(3,3), name='MLPConv0'),
#                               MLPConv(filters=32, kernel_size=(3,3),activation='relu', name='MLPConv1'),
#                               tf.keras.layers.GlobalAvgPool2D(),
#                               tf.keras.layers.Dense(num_classes, activation='softmax', name='DenseSoftmax')
#   ])

model = create_model()
model.compile(optimizer=optimizer, loss=loss_object, metrics=train_metric)
history = model.fit_generator(train_ds, epochs=epochs, validation_data=valid_ds)

loss, acc = test_model(model.layers, train_ds, loss_object,train_metric)
logger.info('Loss={} and accuracy={} for train_ds using original model.'.format(loss, acc))
loss, acc = test_model(model.layers, valid_ds, loss_object,train_metric)
logger.info('Loss={} and accuracy={} for valid_ds using original model.'.format(loss, acc))
loss, acc = test_model(model.layers, test_ds, loss_object,train_metric)
logger.info('Loss={} and accuracy={} for test_ds using original model.'.format(loss, acc))

#Pruned 127 weights and kept same accuracy 1.0.
# compressor = DeepCompression(model, train_ds)
# layers = compressor.compress_layers(['Dense2'], threshold=0.001)

#Shape mismatch, number of filters and number of units in replaced dense must be equal.
# compressor = ReplaceDenseWithGlobalAvgPool(model, train_ds)
# layers = compressor.compress_layers(['Dense2'])

#Correct
# compressor = InsertDenseSVD(model, train_ds)
# layers = compressor.compress_layers(['Dense2'], verbose=1, iterations=10000)

# a veces si a veces no
# compressor = InsertDenseSparse(model, train_ds)
# layers = compressor.compress_layers(['Dense2'], verbose=1, iterations=10000)

#0.67
# compressor = InsertSVDConv(model, train_ds)
# layers = compressor.compress_layers(['conv2d_1'], verbose=1, iterations=20)

#0.5 accuracy Overfit batch training set?
# compressor = DepthwiseSeparableConvolution(model, train_ds)
# layers = compressor.compress_layers(['conv2d_1'], verbose=1, iterations=20)

# Overfit batch training set?
# compressor = FireLayerCompression(model, train_ds)
# layers = compressor.compress_layers(['conv2d_1'], verbose=1, iterations=10)

#0.5 accuracy
compressor = MLPCompression(model, train_ds)
layers = compressor.compress_layers(['conv2d_1'], verbose=1, iterations=5)

# population=20,generations=100 takes 3 hours and 40 minutes
# compressor = SparseConnectionsCompression(model, train_ds, loss_object)
# layers = compressor.compress_layers(['conv2d_1'], population=10,generations=5)#, iterations=100)

loss, acc = test_model(layers, train_ds, loss_object,train_metric)
logger.info('Loss={} and accuracy={} for train_ds using new model.'.format(loss, acc))
loss, acc = test_model(layers, valid_ds, loss_object,train_metric)
logger.info('Loss={} and accuracy={} for valid_ds using new model.'.format(loss, acc))
loss, acc = test_model(layers, test_ds, loss_object,train_metric)
logger.info('Loss={} and accuracy={} for test_ds using new model.'.format(loss, acc))
