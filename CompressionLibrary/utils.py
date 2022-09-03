import tensorflow_datasets as tfds
import tensorflow as tf
from functools import partial
import numpy as np
import pandas as pd
from CompressionLibrary.custom_layers import SparseSVD, SparseConnectionsConv2D, SparseConvolution2D
import tensorflow.keras.backend as K
import gc
import logging



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

def load_dataset(data_path, batch_size=128):
  splits, info = tfds.load('imagenet2012', as_supervised=True, with_info=True, shuffle_files=True, 
                              split=['train[:80%]', 'train[80%:]','validation'], data_dir=data_path)

  (train_examples, validation_examples, test_examples) = splits
  num_examples = info.splits['train'].num_examples

  num_classes = info.features['label'].num_classes
  input_shape = info.features['image'].shape

  input_shape = (224,224,3)

  train_ds = train_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
  valid_ds = validation_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
  test_ds = test_examples.map(imagenet_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

  return train_ds, valid_ds, test_ds, input_shape, num_classes

def calculate_model_weights(model):
    total_weights = 0
    logger = logging.getLogger(__name__)
    for layer in model.layers:
        # Save trainable state.
        trainable_previous_config = layer.trainable
        # Set layer to trainable to calculate number of parameters.
        layer.trainable = True
        
        if 'DeepComp' in layer.name:
            weights, _ = layer.get_weights()
            num_zeroes = tf.math.count_nonzero(tf.abs(weights) == 0.0).numpy()
            weights_before = np.sum([K.count_params(w) for w in layer.trainable_weights])
            weights_after = weights_before - num_zeroes
        elif isinstance(layer, SparseSVD):
            basis, sparse_dict, bias = layer.get_weights()
            non_zeroes_sparse = tf.math.count_nonzero(sparse_dict != 0.0).numpy()
            weights_after = tf.size(basis) + non_zeroes_sparse + tf.size(bias)
        elif isinstance(layer, SparseConnectionsConv2D):
            kernel_size = layer.get_config()['kernel_size']
            connections = layer.get_connections()
            num_zeroes = len(connections) - np.sum(connections)
            if isinstance(kernel_size, int):
                channel_weights = kernel_size**2
            else:
                channel_weights = tf.reduce_prod(kernel_size)
            weights_before = np.sum([K.count_params(w) for w in layer.trainable_weights])
            weights_after = weights_before - (channel_weights * num_zeroes)
        elif isinstance(layer, SparseConvolution2D):
            P, Q, S, bias = layer.get_weights()
            num_zeroes_sparse = tf.math.count_nonzero(S == 0.0).numpy()
            weights_after = tf.size(P) + tf.size(Q) + (tf.size(S) - num_zeroes_sparse) + tf.size(bias)
        else:
            weights_after = np.sum([K.count_params(w) for w in layer.trainable_weights])

        logger.info(f'Layer {layer.name} has {weights_after} weights.')
        total_weights += weights_after
        # Return layer to previous trainable state
        layer.trainable = trainable_previous_config

    return int(total_weights)



def PCA(x, m, high_dim=False):
  # Compute the mean of data matrix.
  x_mean = tf.reduce_mean(x, axis=0)
  # Mean substraction.
  normalized_x = (x - x_mean)

  N, D = x.shape
  # Compute the eigenvectors and eigenvalues of the data covariance matrix.
  if D>N and high_dim:
    cov = tf.linalg.matmul(normalized_x, normalized_x, transpose_b=True) / x.shape[0]
  else:
    cov = tf.linalg.matmul(normalized_x, normalized_x, transpose_a=True) / x.shape[0]
  
  eigvals, eigvecs =   tf.linalg.eigh(cov)
  eigvals = tf.math.real(eigvals)
  eigvecs = tf.math.real(eigvecs)

  if D>N and high_dim:
    eigvecs = tf.linalg.matmul(normalized_x, eigvecs, transpose_a=True)

 
  # Choose the eigenvectors associated with the M largest eigenvalues.
  # Collect these eigenvectors in a matrix B=[b1,...b_M]
  descending_idx = tf.argsort(eigvals,  direction='DESCENDING')
  eigvals = tf.gather(eigvals, descending_idx)
  principal_vals = eigvals[:m]

  eigvecs = tf.transpose(eigvecs)
  eigvecs = tf.gather(indices=descending_idx, params=eigvecs)[:m]
  principal_components = tf.transpose(eigvecs)

  
  B = principal_components
  # Obtain projection matrix 
  P = tf.linalg.matmul(B,B, transpose_b=True)
  reconst = tf.linalg.matmul(normalized_x, P) + x_mean
  
  return reconst, x_mean, principal_vals, principal_components
  
