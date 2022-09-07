from gc import callbacks
import tensorflow as tf
import logging
from CompressionLibrary.utils import load_dataset, extract_model_parts, create_model_from_parts, calculate_model_weights
from CompressionLibrary.CompressionTechniques import *
from CompressionLibrary.custom_callbacks import EarlyStoppingReward

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

def create_model(optimizer, loss, metric):
    model = tf.keras.applications.vgg16.VGG16(
                            include_top=True,
                            weights='imagenet',
                            input_shape=(224,224,3),
                            classes=1000,
                            classifier_activation='softmax'
                        )
    model.compile(optimizer=optimizer, loss=loss,
                    metrics=metric)

    return model

#target_layers = ['block2_conv1', 'block2_conv2']
target_layers = ['fc1']

train_ds, valid_ds, test_ds, input_shape, _ = load_dataset(data_path)

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(1e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    model = create_model(optimizer, loss, metric)
    loss, valid_acc = model.evaluate(valid_ds)

weights_before = calculate_model_weights(model)

optimizer = tf.keras.optimizers.Adam(1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metric = tf.keras.metrics.SparseCategoricalAccuracy()
new_layers = []
for layer_name in target_layers:
    compressor = InsertDenseSparse(model=model, dataset=None, optimizer=optimizer, loss=loss, metrics=metric, input_shape=(224,224,3))
    compressor.compress_layer(layer_name, new_layer_iterations=1200, new_layer_verbose=True)
    model = compressor.get_model()
    new_layers.append(compressor.new_layer_name)


layers, configs, weights = extract_model_parts(model)

with strategy.scope():
    
    optimizer2 = tf.keras.optimizers.Adam(1e-5)
    loss2 = tf.keras.losses.SparseCategoricalCrossentropy()
    metric2 = tf.keras.metrics.SparseCategoricalAccuracy()
    model2 = create_model_from_parts(layers, configs, weights, optimizer2, loss2, metric2)
    model2.compile(optimizer2, loss2, metric2)
    for layer in model2.layers:
        if layer.name in new_layers:
            layer.trainable = True
        else:
            layer.trainable = False
    model.summary()
    model2.summary()
    cb = EarlyStoppingReward(acc_before=valid_acc, weights_before=weights_before, verbose=1)
    model2.fit(train_ds, epochs=10, validation_data=valid_ds, callbacks=[cb])
    loss, acc = model2.evaluate(test_ds)
    weights_after = calculate_model_weights(model2)
    print(f'Model has {weights_after} weights and an accuracy of {acc}.')