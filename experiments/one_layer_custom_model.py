import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds
import logging
import numpy as np

import sys

sys.path.insert(1, '/home/A00806415/DCC/ModelCompression/')

from CompressionLibrary import CompressionTechniques
from CompressionLibrary.utils import calculate_model_weights
from CompressionLibrary.CompressionTechniques import ModelCompression

import importlib
import inspect

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tfds.disable_progress_bar()

current_os = 'linux'
datasets_name = ['fashion_mnist', 'mnist', 'kmnist','horses_or_humans', 'cifar10']
agent_name = 'one_layer_custom_model'


fine_tuning_epochs = 60

if current_os == 'windows':
    data_path = f'G:\\Python projects\\ModelCompressionRL\\data\\'
    log_path = f'G:\\Python projects\\ModelCompressionRL\\data\\logs\\ModelCompression_{agent_name}.log'
    exploration_filename = data_path + f'stats/{agent_name}_training.csv'
    test_filename = data_path + f'stats\\{agent_name}_testing.csv'
    figures_path = data_path+f'figures\\{agent_name}'
    datasets_path = "G:\\ImageNet 2012\\"#data_path+"datasets"
else:
    data_path = '/home/A00806415/DCC/ModelCompression/data'
    log_path = f'/home/A00806415/DCC/ModelCompression/data/logs/ModelCompression_{agent_name}.log'
    exploration_filename = data_path + f'/stats/{agent_name}_training.csv'
    test_filename = data_path + f'/stats/{agent_name}_testing.csv'
    figures_path = data_path+f'/figures/{agent_name}'
    datasets_path = data_path 

logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(log_path, 'w+')],
    format='%(asctime)s -%(levelname)s - %(funcName)s -  %(message)s')
logging.root.setLevel(logging.DEBUG)

log = logging.getLogger('tensorflow')
log.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

logger.info(f'Agent is {agent_name}.')


def prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn, batch_size):
    train_ds = train_examples.shuffle(buffer_size=num_examples).map(map_fn).batch(batch_size)
    valid_ds = validation_examples.map(map_fn).batch(batch_size)
    test_ds = test_examples.map(map_fn).batch(batch_size)
    return train_ds, valid_ds, test_ds


@tf.function
def map_fn(img, label):
    img = tf.image.resize(img, size=(IMAGE_SIZE, IMAGE_SIZE))
    img /= 255.
    return img, label

def create_model(dataset_name, train_ds, valid_ds):
    if current_os =='windows':
        checkpoint_path = f"G:\\Python projects\\ModelCompressionRL\\data\\models\\custom_cnn_{dataset_name}\\cp.ckpt"
    else:
        checkpoint_path = data_path + f"/models/custom_cnn_{dataset_name}/cp.ckpt"

    optimizer = tf.keras.optimizers.Adam(1e-5)
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


    try:
        model.load_weights(checkpoint_path).expect_partial()
    except:
        logging.info(f'No model found for {dataset_name}. Training model...')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)
        model.fit(train_ds,
          epochs=1000,
          validation_data=valid_ds,
          callbacks=[cp_callback], verbose=2)

    return model 


parameters = {}

compressors = [name for name, cls in
               inspect.getmembers(importlib.import_module("CompressionLibrary.CompressionTechniques"), inspect.isclass) if
               issubclass(cls, ModelCompression)]

print(compressors)

# Dense compressors
parameters['DeepCompression'] = {'layer_name': 'dense_0', 'threshold': 0.001}
parameters['ReplaceDenseWithGlobalAvgPool'] = {'layer_name': 'dense_1'}
parameters['InsertDenseSVD'] = {'layer_name': 'dense_0', 'percentage': None, 'hidden_units':None}
parameters['InsertDenseSparse'] = {'layer_name': 'dense_0', 'verbose': True, 'new_layer_iterations':1000, 'new_layer_verbose': True,
                                   'mode':'ksvd'}
# Convolution compressors
parameters['InsertSVDConv'] = {'layer_name': 'conv2d_1'}
parameters['DepthwiseSeparableConvolution'] = {'layer_name': 'conv2d_1'}
parameters['FireLayerCompression'] = {'layer_name': 'conv2d_1'}
parameters['MLPCompression'] = {'layer_name': 'conv2d_1', 'percentage': None, 'hidden_units':None}
parameters['SparseConnectionsCompression'] = {'layer_name': 'conv2d_1', 'epochs':30,
                                              'target_perc':0.75, 'conn_perc_per_epoch':0.1}
parameters['SparseConvolutionCompression'] = {
        'layer_name': 'conv2d_1', 
        'new_layer_iterations': 1000,
        'new_layer_iterations_sparse':100000,
        'new_layer_verbose':True} 


def test_model(test_model, model_name):

    # Calculate stats after compression.
    loss, train_acc = test_model.evaluate(train_ds, verbose=2)
    logger.info('Loss={} and accuracy={} for train_ds using {} model.'.format(loss, train_acc, model_name))
    loss, val_acc = test_model.evaluate(valid_ds, verbose=2)
    logger.info('Loss={} and accuracy={} for valid_ds using {} model.'.format(loss, val_acc, model_name))
    loss, test_acc = test_model.evaluate(test_ds, verbose=2)
    logger.info('Loss={} and accuracy={} for test_ds using {} model.'.format(loss, test_acc, model_name))
    num_parameters = calculate_model_weights(test_model)

    return train_acc, val_acc, test_acc, num_parameters



try:
    df = pd.read_csv(test_filename)
except FileNotFoundError:
    df = pd.DataFrame()

for dataset in datasets_name:


    splits, info = tfds.load(dataset, as_supervised=True, with_info=True,
                            split=['train[:80%]', 'train[80%:]', 'test'], data_dir=data_path+'/datasets/')

    (train_examples, validation_examples, test_examples) = splits

    print(info)
    num_examples = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape
    BATCH_SIZE = 32
    IMAGE_SIZE = None
    input_shape = list(input_shape)
    print(input_shape)
    # Only horses or humans datasets is bigger.
    if input_shape[0]>224 or input_shape[1]>224:
        IMAGE_SIZE = 224
    else:
        IMAGE_SIZE = input_shape[0]

    input_shape[0] = IMAGE_SIZE
    input_shape[1] = IMAGE_SIZE

    print('Number of examples', num_examples)
    print('Number of classes', num_classes) 

    train_ds, valid_ds, test_ds = prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn,
                                              BATCH_SIZE) 
    
    # Loading first time to train model.
    optimizer = optimizer = tf.keras.optimizers.Adam(1e-5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    model = create_model(dataset, train_ds, valid_ds)

    # Calculate stats before compression.

    train_acc_b, val_acc_b, test_acc_b, weights_before = test_model(model, 'original')

    
    for compressor_name in compressors:
        logging.info(f'Using compressor {compressor_name} for dataset {dataset}.')
        # First time to fit in case model is missing.

        tf.keras.backend.clear_session()
        model = create_model(dataset, train_ds, valid_ds)
        print(model.summary())

        class_ = getattr(CompressionTechniques, compressor_name)
        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        cb = None
        

        compressor = class_(model=model, dataset=train_ds, optimizer=optimizer, loss=loss_object, metrics=train_metric, input_shape=input_shape,
                            fine_tuning=False)
        
        # print(df.loc[((df.dataset == dataset)  & (df.compressor==compressor_name))])
        # print(df.loc[((df.dataset == dataset)  & (df.compressor==compressor_name))].shape[0])
        # if df.loc[((df.dataset == dataset)  & (df.compressor==compressor_name))].shape[0]>=1:
        #     continue
        
        if compressor_name not in parameters:
            continue

        compressor_params = parameters[compressor_name]
        compressor.compress_layer(**compressor_params)

        compressed_model = compressor.get_model()

        if compressor_name in ['SparseConnectionsCompression']:
            cb = compressor.callbacks
        

        # Calculate stats after compression.
        train_acc_after, val_acc_after, test_acc_after, weights_after = test_model(compressed_model, 'original')

        info = {}
        info['dataset'] = dataset
        info['compressor'] = compressor_name
        info['fine-tuned layers'] = 'none'
        info['fine-tuning epochs'] = 0
        info['train_acc_before'] = train_acc_b
        info['train_acc_after'] = train_acc_after
        info['val_acc_before'] = val_acc_b
        info['val_acc_after'] = val_acc_after
        info['test_acc_before'] = test_acc_b
        info['test_acc_after'] = test_acc_after
        info['parameters_before'] = weights_before
        info['parameters_after'] = weights_after

        new_row = pd.DataFrame(info, index=[0])
        if not os.path.isfile(test_filename):
            new_row.to_csv(test_filename, index=False)
        else: # else it exists so append without writing the header
            new_row.to_csv(test_filename, mode='a', index=False, header=False)


        fine_tune_all_model = tf.keras.models.clone_model(compressed_model)
        for layer in fine_tune_all_model.layers:
            layer.trainable = True

        print(fine_tune_all_model.summary())
        fine_tune_all_model.compile(optimizer=optimizer, loss=loss_object, metrics=train_metric)
        if compressor_name in ['SparseConnectionsCompression']:
            fine_tune_all_model.fit(train_ds, epochs=30, validation_data=valid_ds, verbose=2, callbacks=cb)
        else: 
            fine_tune_all_model.fit(train_ds, epochs=30, validation_data=valid_ds, verbose=2)

        # Calculate stats after fine-tuning all layers.
        train_acc_after, val_acc_after, test_acc_after, weights_after = test_model(fine_tune_all_model, 'fine-tuning all')
        info['fine-tuned layers'] = 'all'
        info['fine-tuning epochs'] = 30
        info['train_acc_after'] = train_acc_after
        info['val_acc_after'] = val_acc_after
        info['test_acc_after'] = test_acc_after
        info['parameters_after'] = weights_after

        new_row = pd.DataFrame(info, index=[0])
        if not os.path.isfile(test_filename):
            new_row.to_csv(test_filename, index=False)
        else: # else it exists so append without writing the header
            new_row.to_csv(test_filename, mode='a', index=False, header=False)



        fine_tune_compressed_model = tf.keras.models.clone_model(compressed_model)
        for layer in fine_tune_compressed_model.layers:
            if ('conv2d_1' in layer.name and compressor.target_layer_type=='conv') or ('dense_0' in layer.name and compressor.target_layer_type=='dense') or ('GAP' in layer.name):
                layer.trainable = True
            else:
                layer.trainable = False
        
        print(fine_tune_compressed_model.summary())
        fine_tune_compressed_model.compile(optimizer=optimizer, loss=loss_object, metrics=train_metric)

        if compressor_name in ['SparseConnectionsCompression']:
            fine_tune_compressed_model.fit(train_ds, epochs=30, validation_data=valid_ds, verbose=2, callbacks=cb)
        else:
            fine_tune_compressed_model.fit(train_ds, epochs=30, validation_data=valid_ds, verbose=2)
        
        # Calculate stats after fine-tuning compressed layers.
        train_acc_after, val_acc_after, test_acc_after, weights_after = test_model(fine_tune_compressed_model, 'fine-tuning compressed')
        info['fine-tuned layers'] = 'compressed'
        info['fine-tuning epochs'] = 30
        info['train_acc_after'] = train_acc_after
        info['val_acc_after'] = val_acc_after
        info['test_acc_after'] = test_acc_after
        info['parameters_after'] = weights_after

        new_row = pd.DataFrame(info, index=[0])
        if not os.path.isfile(test_filename):
            new_row.to_csv(test_filename, index=False)
        else: # else it exists so append without writing the header
            new_row.to_csv(test_filename, mode='a', index=False, header=False)
