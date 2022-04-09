import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds
import logging
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile
from tensorflow import keras
import utils

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

def count_zeroes_in_dense_layer(layer):

    print(tf.math.count_nonzero(layer._non_trainable_weights[0]== 0.0))
    print(tf.math.count_nonzero(layer._non_trainable_weights[0]== 0))
    return tf.math.count_nonzero(layer._non_trainable_weights[0]== 0.0).numpy()

def get_model():

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_0',
                                                        input_shape=input_shape),
                                 tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
                                 tf.keras.layers.MaxPool2D((2, 2), 2),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(128, activation='relu', name='dense_0'),
                                 tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
                                 tf.keras.layers.Dense(num_classes, activation='softmax', name='dense_softmax')
                                 ])
    model.compile(optimizer=optimizer, loss=loss_object, metrics=train_metric)
    path = './data/saved_models/{}/'.format(dataset_name)

    # model.build(input_shape=input_shape)
    model.load_weights(path)
    return model

# horses_or_humans
# mnist
# cifar10
# imagenet2021
# fashion_mnist

datasets = ['horses_or_humans', 'mnist', 'cifar10', 'fashion_mnist']
dataset_name = 'cifar10'

df = pd.DataFrame()
for dataset_name in datasets:
    train_ds, val_ds, test_ds, input_shape, num_classes = utils.load_dataset(dataset_name)



    model = get_model()

    loss, train_acc_b = model.evaluate(train_ds)
    logger.info('Loss={} and accuracy={} for train_ds using original model.'.format(loss, train_acc_b))
    loss, val_acc_b = model.evaluate(val_ds)
    logger.info('Loss={} and accuracy={} for valid_ds using original model.'.format(loss, val_acc_b))
    loss, test_acc_b = model.evaluate(test_ds)
    logger.info('Loss={} and accuracy={} for test_ds using original model.'.format(loss, test_acc_b))


    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    batch_size = 128
    epochs = 5
    validation_split = 0.1

    num_images = len(train_ds)
    end_step = np.ceil(num_images/batch_size).astype(np.int32)*epochs

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.5,
                                                                 final_sparsity=0.8,
                                                                 begin_step=0,
                                                                end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    model_for_pruning.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])

    logdir = tempfile.mkdtemp()

    callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]


    model_for_pruning.fit(train_ds,
                      batch_size=batch_size, epochs=epochs, validation_data=val_ds,
                      callbacks=callbacks)

    loss, train_acc_b = model_for_pruning.evaluate(train_ds)
    logger.info('Loss={} and accuracy={} for train_ds using original model.'.format(loss, train_acc_b))
    loss, val_acc_b = model_for_pruning.evaluate(val_ds)
    logger.info('Loss={} and accuracy={} for valid_ds using original model.'.format(loss, val_acc_b))
    loss, test_acc_b = model_for_pruning.evaluate(test_ds)
    logger.info('Loss={} and accuracy={} for test_ds using original model.'.format(loss, test_acc_b))

    df_temp = pd.read_csv('./data/results_{}.csv'.format(dataset_name))
    df_temp['poli_decay_pruning_train_acc'] = train_acc_b
    df_temp['poli_decay_pruning_val_acc'] = val_acc_b
    df_temp['poli_decay_pruning_test_acc'] = test_acc_b

    print(model.summary())
    print(model_for_pruning.summary())
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    zeroes = 0
    for layer in model_for_pruning.layers[4:]:
        print(layer._non_trainable_weights)
        print(layer.name)
        zeroes += count_zeroes_in_dense_layer(layer)
    print('Pruned model has {} zeroes'.format(zeroes))
    zeroes += 0
    print(model_for_export.summary())
    for layer in model_for_export.layers[4:]:
        print(layer.name)
        print(tf.math.count_nonzero(layer.get_weights()[0] == 0.0))
        print(tf.math.count_nonzero(layer.get_weights()[0] == 0))
        zeroes += tf.math.count_nonzero(layer.get_weights()[0] == 0.0).numpy()

    print('Pruned model has {} zeroes'.format(zeroes))
    df_temp['weights_after_prune'] = df_temp['weights_before'] - zeroes
    df = df.append(df_temp)


print(df.to_string())

