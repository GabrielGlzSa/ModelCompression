import tensorflow as tf
import tensorflow_datasets as tfds
import logging

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
                            split=['train[:80%]', 'train[80%:]','validation'], data_dir=data_path)

(train_examples, validation_examples, test_examples) = splits
num_examples = info.splits['train'].num_examples

num_classes = info.features['label'].num_classes
input_shape = info.features['image'].shape

input_shape = (224,224,3)
batch_size = 32

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
  return model



with strategy.scope():
    model = create_model()
    loss, acc_before = model.evaluate(valid_ds)


print(f'Validation accuracy of {acc_before} and {loss} loss.')