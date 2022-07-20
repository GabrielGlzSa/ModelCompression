import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import logging
from CompressionLibrary.utils import load_dataset
from CompressionLibrary.CompressionTechniques import *

import time

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# tf.get_logger().setLevel('ERROR')

# logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logger = logging.getLogger(__name__)

tfds.disable_progress_bar()

dataset = 'cifar10'
train_ds, valid_ds, test_ds, input_shape, num_classes = load_dataset(dataset)

optimizer = tf.keras.optimizers.Adam(1e-5)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

epochs = 1


def create_model(input_shape):
    model_path = './data/models/vgg16/test_'+dataset
    try:
        model = tf.keras.models.load_model(model_path, compile=True)
    except:

        model = tf.keras.applications.vgg16.VGG16(
                    include_top=True,
                    weights=None,
                    input_shape=input_shape,
                    classes=num_classes,
                    classifier_activation='softmax'
                )
        model.compile(optimizer=optimizer, loss=loss_object, metrics=train_metric)
        history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds)

    
    return model


model = create_model(input_shape)
model.summary()
# print(model.evaluate(train_ds))

start_time = time.time()
# Pruned 127 weights and kept same accuracy 1.0.
# compressor = DeepCompression(model=model, dataset=train_ds,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
# compressor.compress_layer('fc1')

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

compressor = FireLayerCompression(model=model, dataset=train_ds,
                                  optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
compressor.compress_layer('block2_conv1')

# compressor = MLPCompression(model=model, dataset=train_ds,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)

# compressor.compress_layer('block2_conv1')


# compressor = SparseConnectionsCompressionCustom(model=model,
#                                                 dataset=train_ds,
#                                                 optimizer=optimizer,
#                                                 loss=loss_object,
#                                                 metrics=train_metric)
# compressor.compress_layer('block2_conv1', epochs=10)

# RAM OOM in training loop.
# compressor = SparseConvolutionCompression(model=model, dataset=train_ds,
#                             optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape)
# compressor.compress_layer('block2_conv1', iterations=2, bases=32)

print("--- %s seconds ---" % (time.time() - start_time))

new_model = compressor.get_model()
new_model.summary()

new_model.evaluate(train_ds)