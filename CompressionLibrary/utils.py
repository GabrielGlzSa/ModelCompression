import tensorflow_datasets as tfds
import tensorflow as tf
from functools import partial
import numpy as np
import pandas as pd
import CompressionLibrary.environments as env_lib
import gc


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


@tf.function
def map_fn(img, label, img_size):
    img = tf.image.resize(img, size=(img_size, img_size))
    img /= 255.
    return img, label


def load_dataset(dataset_name='mnist'):
    splits, info = tfds.load(dataset_name, as_supervised=True, with_info=True,
                             split=['train[:80%]', 'train[80%:]', 'test'], data_dir='./data')

    (train_examples, validation_examples, test_examples) = splits
    num_examples = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes
    input_shape = info.features['image'].shape
    BATCH_SIZE = 32
    input_shape = list(input_shape)
    if input_shape[0] > 224:
        img_size = 224
    else:
        img_size = input_shape[0]

    input_shape[0] = img_size
    input_shape[1] = img_size

    partial_map_fn = partial(map_fn, img_size=img_size)
    train_ds, val_ds, test_ds = prepare_dataset(train_examples, validation_examples, test_examples, num_examples, partial_map_fn,
                                                BATCH_SIZE)
    return train_ds, val_ds, test_ds, input_shape, num_classes


def prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn, batch_size):
    train_ds = train_examples.shuffle(
        buffer_size=num_examples).map(map_fn).batch(batch_size)
    valid_ds = validation_examples.map(map_fn).batch(batch_size)
    test_ds = test_examples.map(map_fn).batch(batch_size)

    return train_ds, valid_ds, test_ds


def evaluate_adadeep(env, conv_agent, fc_agent, n_samples_mode, reward_func, n_games=1):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    acc = []
    weights = []
    infos = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        df = pd.DataFrame()
        for k in range(len(env.layer_name_list)):
            next_layer_name = env.layer_name_list[env._layer_counter]
            layer = env.model.get_layer(next_layer_name)
            was_conv = False
            if n_samples_mode < s.shape[0]:
                random_images = s[np.random.choice(
                    s.shape[0], size=n_samples_mode, replace=False)]
            else:
                random_images = s
            if isinstance(layer, tf.keras.layers.Conv2D):
                # qvalues = conv_agent.get_qvalues(random_image).numpy()
                # action = conv_agent.sample_actions(qvalues)[0]
                qvalues = conv_agent.get_qvalues(random_images).numpy()
                action = conv_agent.sample_actions_using_mode(qvalues)[0]
                was_conv = True
            if isinstance(layer, tf.keras.layers.Dense):
                qvalues = fc_agent.get_qvalues(random_images).numpy()
                action = fc_agent.sample_actions_using_mode(qvalues)[0]

            new_s, r, done, info = env.step(action)
            s = env.get_state('current_state')
            if done:
                s = env.reset()
                break

            row = {'state': s, 'action': action, 'reward': r,
                   'next_state': new_s, 'done': done, 'info': info}
            df = df.append(row, ignore_index=True)

        # Calculate reward using stats before and after compression
        before_stats = df.iloc[0]['info']
        after_stats = df.iloc[-1]['info']
        reward = reward_func(before_stats, after_stats)

        rewards.append(reward)
        acc.append(info['acc_after'])
        weights.append(info['weights_after'])
    return np.mean(rewards), np.mean(acc), np.mean(weights)


def play_and_record_adadeep(conv_agent, fc_agent, env, conv_replay, fc_replay, dataset, current_state, next_state, n_samples_mode, reward_func, n_steps=1, save_replay=False, debug=False):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time
    """
    # initial state
    s = env.reset()
    # Play the game for n_steps as per instructions above
    rewards = 0

    for it in range(n_steps):
        df = pd.DataFrame()
        for k in range(1, len(env.layer_name_list)+1):
            tf.keras.backend.clear_session()

            current_layer_name = env.layer_name_list[env._layer_counter]
            layer = env.model.get_layer(current_layer_name)
            was_conv = False
            if n_samples_mode < s.shape[0]:
                random_images = s[np.random.choice(
                    s.shape[0], size=n_samples_mode, replace=False)]
            else:
                random_images = s
            if isinstance(layer, tf.keras.layers.Conv2D):
                # qvalues = conv_agent.get_qvalues(random_image).numpy()
                # action = conv_agent.sample_actions(qvalues)[0]
                qvalues = conv_agent.get_qvalues(random_images).numpy()
                action = conv_agent.sample_actions_using_mode(qvalues)[0]
                was_conv = True
            if isinstance(layer, tf.keras.layers.Dense):
                qvalues = fc_agent.get_qvalues(random_images).numpy()
                action = fc_agent.sample_actions_using_mode(qvalues)[0]

            new_s, r, done, info = env.step(action)

            info['was_conv'] = was_conv

            if debug:
                print('Iteration {} - Layer {}/{}'.format(it, k, len(env.layer_name_list)),
                      s.shape, action, r, new_s.shape, done, info)

            row = {'state': s, 'action': action, 'reward': r,
                   'next_state': new_s, 'done': done, 'info': info}
            df = df.append(row, ignore_index=True)

            s = env.get_state('current_state')

            if done:
                s = env.reset()
                break
            gc.collect()

        # Calculate reward using stats before and after compression
        before_stats = df.iloc[0]['info']
        after_stats = df.iloc[-1]['info']

        reward = 1 - after_stats['weights_after'] / \
            before_stats['weights_before'] + after_stats['acc_after']

        # Set the same reward for all actions.
        df['reward'] = reward

        for idx, row in df.iterrows():
            was_conv = row['info']['was_conv']
            if was_conv:
                for idx, feature in enumerate(row['state']):
                    conv_replay.add(
                        feature, row['action'], row['reward'], row['next_state'][idx], row['done'])
            else:
                for idx, feature in enumerate(row['state']):
                    fc_replay.add(
                        feature, row['action'], row['reward'], row['next_state'][idx], row['done'])
        if save_replay:
            fc_replay.save(
                './data/{}_adadeep_fc_replay_{}_{}.pkl'.format(dataset, current_state, next_state))
            conv_replay.save(
                './data/{}_adadeep_conv_replay_{}_{}.pkl'.format(dataset, current_state, next_state))
    return reward


def make_env(dataset, layer_name_list, current_state_source='layer_input', next_state_source='layer_output', base_model='vgg16', first_time_epochs=100):
    train_ds, valid_ds, test_ds, input_shape, num_classes = load_dataset(
        dataset)

    model_path = './data/models/'+base_model+'/test_'+dataset
    try:
        model = tf.keras.models.load_model(model_path, compile=True)
    except OSError:
        optimizer = tf.keras.optimizers.Adam(1e-5)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        if base_model == 'simple_custom':
            model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_0',
                                                                input_shape=input_shape),
                                        tf.keras.layers.Conv2D(
                                            32, (3, 3), activation='relu', name='conv2d_1'),
                                        tf.keras.layers.MaxPool2D((2, 2), 2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(
                                            128, activation='relu', name='dense_0'),
                                        tf.keras.layers.Dense(
                                            128, activation='relu', name='dense_1'),
                                        tf.keras.layers.Dense(
                                            num_classes, activation='softmax', name='dense_softmax')
                                        ])
        elif base_model == 'vgg16':
            model = tf.keras.applications.vgg16.VGG16(
                include_top=True,
                weights=None,
                input_shape=input_shape,
                classes=num_classes,
                classifier_activation='softmax'
            )
        else:
            raise "No valid model was passed. Expected 'simple_custom' or 'vgg16'."

        model.compile(optimizer=optimizer, loss=loss_object,
                      metrics=train_metric)
        model.fit(train_ds, epochs=first_time_epochs, validation_data=valid_ds, verbose=0)
        model.save(model_path)

    w_comprs = ['InsertDenseSVD', 'InsertDenseSparse',
                'DeepCompression']  # 'InsertDenseSVDCustom'
    c_comprs = ['InsertSVDConv', 'SparseConvolutionCompression',
                'DepthwiseSeparableConvolution', 'SparseConnectionsCompression']
    l_comprs = ['FireLayerCompression', 'MLPCompression',
                'ReplaceDenseWithGlobalAvgPool']
    compressors_list = w_comprs + c_comprs + l_comprs

    parameters = {}
    parameters['DeepCompression'] = {
        'layer_name': 'dense_0', 'threshold': 0.001}
    parameters['ReplaceDenseWithGlobalAvgPool'] = {'layer_name': 'dense_1'}
    parameters['InsertDenseSVD'] = {'layer_name': 'dense_0', 'units': 16}
    parameters['InsertDenseSVDCustom'] = {'layer_name': 'dense_0', 'units': 16}
    parameters['InsertDenseSparse'] = {
        'layer_name': 'dense_0', 'verbose': True, 'units': 16}
    parameters['InsertSVDConv'] = {'layer_name': 'conv2d_1', 'units': 8}
    parameters['DepthwiseSeparableConvolution'] = {'layer_name': 'conv2d_1'}
    parameters['FireLayerCompression'] = {'layer_name': 'conv2d_1'}
    parameters['MLPCompression'] = {'layer_name': 'conv2d_1'}
    parameters['SparseConvolutionCompression'] = {
        'layer_name': 'conv2d_1', 'bases': 4}
    parameters['SparseConnectionsCompression'] = {'layer_name': 'conv2d_1', 'epochs': 20,
                                                  'target_perc': 0.75, 'conn_perc_per_epoch': 0.1}

    env = env_lib.ModelCompressionEnv(compressors_list, model_path, parameters,
                                      train_ds, valid_ds, test_ds,
                                      layer_name_list, input_shape, current_state_source=current_state_source, next_state_source=next_state_source, verbose=False)

    return env


if __name__ == '__main__':
    dataset = load_dataset('horses_or_humans')
