import tensorflow as tf
import tensorflow_datasets as tfds
import logging
from CompressionLibrary.utils import load_dataset, create_model

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
  raise('ERROR: Not connected to a TPU runtime; Using GPU strategy instead!')
  



with strategy.scope():
  train_ds, valid_ds, test_ds, input_shape, num_classes = create_dataset(batch_size=32*8)
  model = create_model(num_classes)
  loss, acc_before = model.evaluate(valid_ds)
  print(f'Validation accuracy of {acc_before} and {loss} loss.')
  loss, acc_before = model.evaluate(test_ds)
  print(f'Test accuracy of {acc_before} and {loss} loss.')