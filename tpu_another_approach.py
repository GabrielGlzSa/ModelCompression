import tensorflow as tf
import logging
from CompressionLibrary.utils import load_dataset
from CompressionLibrary.CompressionTechniques import MLPCompression

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
batch_size_per_replica = 128
tuning_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

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

target_layers = ['block2_conv1', 'block2_conv2']

with strategy.scope():
    train_ds, valid_ds, input_shape, batch_size = load_dataset()
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    model = create_model()
    model.compile(optimizer, loss, metric)
    for layer_name in target_layers:
        compressor = MLPCompression(model=model, dataset=train_ds, optimizer=optimizer, loss=loss, metrics=metric, input_shape=input_shape)

        new_layer, new_layer_name, weights_before, weights_after = compressor.get_new_layer(model.get_layer(layer_name))

        model = compressor.replace_layer(new_layer, layer_name)
        model.compile(optimizer, loss, metric)
    model.fit(train_ds, epochs=10, validation_data=valid_ds)
