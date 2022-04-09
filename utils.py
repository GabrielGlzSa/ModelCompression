import tensorflow_datasets as tfds
import tensorflow as tf
from functools import partial

@tf.function
def map_fn(img, label, img_size):
    img = tf.image.resize(img, size=(img_size, img_size))
    img /= 255.
    return img, label

def load_dataset(dataset_name = 'horses_or_humans'):
    splits, info = tfds.load(dataset_name, as_supervised=True, with_info=True,
                             split=['train[:80%]', 'train[80%:]', 'test'], data_dir='./data')

    (train_examples, validation_examples, test_examples) = splits

    # print(info)
    num_examples = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape
    BATCH_SIZE = 32
    input_shape = list(input_shape)
    # print(input_shape)
    if input_shape[0]>224:
        img_size = 224
    else:
        img_size = input_shape[0]

    input_shape[0] = img_size
    input_shape[1] = img_size

    # print('Number of examples', num_examples)
    # print('Number of classes', num_classes)

    partial_map_fn = partial(map_fn, img_size=img_size)
    train_ds, val_ds, test_ds = prepare_dataset(train_examples, validation_examples, test_examples, num_examples, partial_map_fn,
                                              BATCH_SIZE)
    return train_ds, val_ds, test_ds, input_shape, num_classes





def prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn, batch_size):
    train_ds = train_examples.shuffle(buffer_size=num_examples).map(map_fn).batch(batch_size)
    valid_ds = validation_examples.map(map_fn).batch(batch_size)
    test_ds = test_examples.map(map_fn).batch(batch_size)

    return train_ds, valid_ds, test_ds


if __name__=='__main__':
    dataset = load_dataset('horses_or_humans')
