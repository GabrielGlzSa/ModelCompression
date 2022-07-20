# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59) 
# [GCC 7.5.0]
# Embedded file name: ./ModelCompression/CompressionTechniques.py
# Compiled at: 2022-05-16 17:17:52
# Size of source mod 2**32: 57841 bytes
import math, tensorflow as tf, tensorflow.keras.backend as K, numpy as np, logging, copy, random, re, sys

class ModelCompression:
    __doc__ = '\n    Base class for compressing a deep learning model. The class takes a tensorflow\n    model and a dataset that will be used to fit a regression.\n    '

    def __init__(self, model, optimizer, loss, metrics, input_shape, dataset, fine_tune=True, tuning_epochs=2, verbose=1):
        """

        :param model: tensorflow model that will be optimized.
        :param optimizer: optimizer that will be used to compile the optimized model.
        :param loss: loss object that will be used to compile the optimized model.
        :param metrics: metrics that will be used to compile the optimized model.
        :param dataset: dataset that will be used for compressing a layer and/or fine-tuning.
        :param fine_tune: flag to select if the optimized model will be trained.
        :param tuning_epochs: number of epochs of the fine-tuning.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss
        self.metrics = metrics
        self.dataset = dataset
        self.fine_tune = fine_tune
        self.input_shape = input_shape
        self.tuning_epochs = tuning_epochs
        self.logger = logging.getLogger(__name__)
        self.model_changes = {}
        self.weights_before = self.count_trainable_weights()
        self.weights_after = None
        self.target_layer_type = None
        self.fit_verbose = verbose
        self.new_layer_name = None

    def fine_tune_model(self):
        self.logger.info('Fine-tuning the whole optimized model.')
        self.model.fit(self.dataset, epochs=self.tuning_epochs, verbose=self.fit_verbose)

    def get_technique(self):
        """
        Returns the name of the compression technique.
        :return: name of the class.
        """
        return self.__class__.__name__

    def count_trainable_weights(self):
        """
        Calculate the number of trainable weights in a deep learning model.
        :return: Number of trainable weights in self.model.
        """
        return int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))

    def get_weights_diff(self):
        """
        Get the number of trainable weights before and after compression.
        :return: weights before compression and after compression.
        """
        return (
         self.weights_before, int(self.weights_after))

    def find_layer(self, layer_name: str) -> int:
        """
        Returns the index of the layer in model.layers.
        :param layer_name: name of the layer.
        :return: index of the layer.
        """
        names = [layer.name for layer in self.model.layers]
        return names.index(layer_name)

    def get_model(self):
        """
        Returns the model, which can be the original model or the compressed model if the original model has already
        been compressed.
        :return: Keras model.
        """
        return self.model

    def compress_layer(self, layer_name: str):
        pass


class DeepCompression(ModelCompression):
    __doc__ = '\n    Compression technique that sets to 0 all weights that are below a threshold in\n    a Dense layer. No clustering and Huffman encoding is performed.\n    '

    def __init__(self, **kwargs):
        (super(DeepCompression, self).__init__)(**kwargs)
        self.target_layer_type = 'dense'

    def compress_layer(self, layer_name: str, threshold: float=0.0001):
        self.logger.info('Searching for layer: {}'.format(layer_name))
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        weights = layer.get_weights()
        tf_weights = tf.Variable((weights[0]), trainable=True)
        below_threshold = tf.abs(tf_weights) < threshold
        self.logger.info('Pruned weights: {}'.format(tf.math.count_nonzero(below_threshold)))
        new_weights = tf.where(below_threshold, 0.0, tf_weights)
        layer.set_weights([new_weights, weights[1]])
        num_zeroes = tf.math.count_nonzero(tf.abs(self.model.layers[idx].get_weights()[0]) == 0.0).numpy()
        self.logger.debug('Found {} weights with value 0.0'.format(num_zeroes))
        self.logger.info('Finished compression')
        self.weights_after = self.weights_before - num_zeroes
        layer._name = layer.name + '/DeepCompression'
        self.new_layer_name = layer.name


class ReplaceDenseWithGlobalAvgPool(ModelCompression):
    __doc__ = '\n    Compression technique that replaces all dense and flatten layers\n    between the last convolutional layer and the softmax layer with a GlobalAveragePooling2D layer.\n    '

    def __init__(self, **kwargs):
        (super(ReplaceDenseWithGlobalAvgPool, self).__init__)(**kwargs)
        self.target_layer_type = 'dense'

    def compress_layer(self, layer_name, iterations=5):
        self.logger.info('Searching for all dense layers.')
        regexp = re.compile('dense|fc|flatten')
        filters = None
        removed_weights = 0
        softmax_weights = None
        first_non_input = 0
        inputs = tf.keras.layers.Input(shape=(self.input_shape))
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            config = layer.get_config()
            lname = config['name'].lower()
            layer.trainable = False
            if regexp.search(lname):
                if 'softmax' in lname:
                    x = tf.keras.layers.GlobalAveragePooling2D(name=(layer_name + '/GlobalAvgPool'))(x)
                    num_classes = config['units']
                    softmax_weights = layer.get_weights()[0]
                    softmax = tf.keras.layers.Dense(num_classes, input_shape=[
                     filters],
                      activation='softmax',
                      name='softmax')
                    x = softmax(x)
                    layer_removed_weights = np.sum([K.count_params(w) for w in layer.trainable_weights])
                    softmax_weights = np.sum([K.count_params(w) for w in softmax.trainable_weights])
                    self.logger.debug('Replaced {} with {} weights with {} weights.'.format(layer.name, layer_removed_weights, softmax_weights))
                    removed_weights += layer_removed_weights
                else:
                    layer_removed_weights = np.sum([K.count_params(w) for w in layer.trainable_weights])
                    self.logger.debug('Removed {} with {} weights.'.format(layer.name, layer_removed_weights))
                    removed_weights += layer_removed_weights
            else:
                if 'filters' in config.keys():
                    x = layer(x)
                    filters = config['filters']

        self.model = tf.keras.Model(inputs, x)
        self.model.compile(optimizer=(self.optimizer), loss=(self.loss_object),
          metrics=(self.metrics))
        self.model.fit((self.dataset), epochs=iterations, verbose=(self.fit_verbose))
        for layer in self.model.layers:
            layer.trainable = True

        self.new_layer_name = layer_name + '/GlobalAvgPool'
        if self.fine_tune:
            self.fine_tune_model()
        self.weights_after = self.weights_before - removed_weights + softmax_weights
        self.logger.info('Finished compression')


class DenseSVD(tf.keras.Model):

    def __init__(self, units0, units1, weights0, weights1, bias1, name, activation, **kwargs):
        (super(DenseSVD, self).__init__)(name, **kwargs)
        self.dense0 = tf.keras.layers.Dense(units0, weights=[
         weights0],
          activation=(tf.keras.activations.linear),
          use_bias=False,
          name=(name + '/dense_0'))
        self.dense1 = tf.keras.layers.Dense(units1,
          weights=[weights1, bias1], activation=activation, name=(name + '/dense_1'))

    def call(self, inputs):
        x = self.dense0(inputs)
        x = self.dense1(x)
        return x


class InsertDenseSVD(ModelCompression):
    __doc__ = '\n    Compression technique that inserts a smaller dense layer using Singular Value Decomposition. By\n    inserting a smaller layer with C units, the number of weights is reduced\n    from MxN to (M+N)xC. The weights are obtained by fitting a neural network to\n    predict the same output as the original model.\n    '

    def __init__(self, **kwargs):
        (super(InsertDenseSVD, self).__init__)(**kwargs)
        self.target_layer_type = 'dense'

    def compress_layer(self, layer_name, units, iterations=5):
        self.logger.info('Searching for layer: {}'.format(layer_name))
        self.layer_name = layer_name
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        weights, bias = layer.get_weights()
        weights = tf.Variable(weights, dtype='float32')
        s, u, v = tf.linalg.svd(weights, full_matrices=True)
        u = tf.slice(u, begin=[0, 0], size=[u.shape[0], units])
        s = tf.slice(s, begin=[0], size=[units])
        v = tf.slice(v, begin=[0, 0], size=[units, v.shape[1]])
        n = tf.matmul(tf.linalg.diag(s), v)
        loss = tf.reduce_mean(tf.nn.l2_loss(weights - tf.matmul(u, n)))
        self.logger.debug('New weights have {} L2 loss mean.'.format(loss))
        inputs = tf.keras.layers.Input(shape=(self.input_shape))
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            if layer.name == layer_name:
                new_layer = DenseSVD(units0=units, units1=(weights.shape[(-1)]), weights0=u, weights1=n, bias1=bias,
                  activation=(layer.get_config()['activation']),
                  name=(layer_name + '/DenseSVD'))
                new_layer._name = layer_name + '/DenseSVD'
                x = new_layer(x)
            else:
                layer.trainable = False
                x = layer(x)

        self.model = tf.keras.Model(inputs, x)
        self.model.compile(optimizer=(self.optimizer), loss=(self.loss_object),
          metrics=(self.metrics))
        self.model.fit((self.dataset), epochs=(self.tuning_epochs), verbose=(self.fit_verbose))
        for layer in self.model.layers:
            layer.trainable = True

        self.new_layer_name = layer_name + '/DenseSVD'
        if self.fine_tune:
            self.fine_tune_model()
        self.weights_after = self.weights_before - (tf.size(weights) - (tf.size(u) + tf.size(n)))
        self.logger.info('Finished compression')


class InsertDenseSVDCustom(ModelCompression):
    __doc__ = '\n    Compression technique that inserts a smaller dense layer to reduce the number of weights. By\n    inserting a smaller layer with C units, the number of weights is reduced\n    from MxN to (M+N)xC. The weights are obtained by fitting a neural network to\n    predict the same output as the original model.\n    '

    def __init__(self, **kwargs):
        (super(InsertDenseSVDCustom, self).__init__)(**kwargs)
        self.target_layer_type = 'dense'

    def compress_layer(self, layer_name, units, iterations=10):
        self.logger.info('Searching for layer: {}'.format(layer_name))
        self.layer_name = layer_name
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        weights, bias = layer.get_weights()
        weights = tf.Variable(weights, dtype='float32')
        w_init = tf.random_normal_initializer()
        w1 = tf.Variable(name='w1', initial_value=w_init(shape=(weights.shape[0], units)))
        w2 = tf.Variable(name='w2', initial_value=w_init(shape=(units, weights.shape[1])))
        optimizer = tf.keras.optimizers.Adam()
        for i in range(iterations):
            with tf.GradientTape() as (tape):
                pred = tf.matmul(w1, w2)
                loss = tf.reduce_sum(tf.nn.l2_loss(weights - pred))
            gradients = tape.gradient(loss, [w1, w2])
            optimizer.apply_gradients(zip(gradients, [w1, w2]))
            self.logger.debug('Epoch {} Loss: {}'.format(i, loss))

        self.logger.debug('Changing weights from {} to {}+{}.'.format(weights.shape, w1.shape, w2.shape))
        inputs = tf.keras.layers.Input(shape=(self.input_shape))
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            if layer.name == layer_name:
                new_layer = DenseSVD(units0=units, units1=(weights.shape[(-1)]), weights0=(w1.numpy()), weights1=(w2.numpy()), bias1=bias,
                  activation=(layer.get_config()['activation']),
                  name=(layer_name + '/DenseSVDC'))
                new_layer._name = layer_name + '/DenseSVDC'
                x = new_layer(x)
            else:
                layer.trainable = False
                x = layer(x)

        self.model = tf.keras.Model(inputs, x)
        self.model.compile(optimizer=(self.optimizer), loss=(self.loss_object),
          metrics=(self.metrics))
        self.model.fit((self.dataset), epochs=(self.tuning_epochs), verbose=(self.fit_verbose))
        for layer in self.model.layers:
            layer.trainable = True

        self.new_layer_name = layer_name + '/DenseSVDC'
        if self.fine_tune:
            self.fine_tune_model()
        self.weights_after = self.weights_before - (tf.size(weights) - (tf.size(w1) + tf.size(w2)))
        self.logger.info('Finished compression')


class BinaryWeight(tf.keras.constraints.Constraint):
    __doc__ = '\n    Contraint that rounds the weights to the closest value between 0 and 1.\n    '

    def __init__(self, max_value):
        super(BinaryWeight, self).__init__()
        self.max_value = max_value

    def __call__(self, weights):
        return tf.round(tf.clip_by_norm(weights, (self.max_value), axes=0))


class DenseSparseSVD(tf.keras.Model):

    def __init__(self, units0, units1, weights0, weights1, bias1, name, activation, **kwargs):
        (super(DenseSparseSVD, self).__init__)(name, **kwargs)
        self.dense0 = tf.keras.layers.Dense(units0, weights=[
         weights0],
          activation=(tf.keras.activations.linear),
          use_bias=False,
          name=(name + '/dense_0'))
        self.dense1 = tf.keras.layers.Dense(units1, weights=[
         weights1, bias1],
          activation=activation,
          name=(name + '/dense_1'),
          kernel_constraint=(BinaryWeight(units0)))

    def call(self, inputs):
        x = self.dense0(inputs)
        x = self.dense1(x)
        return x


class InsertDenseSparse(ModelCompression):
    __doc__ = '\n    Compression technique that inserts a Dense layer inbetween two Dense layers.\n    By inserting a smaller layer with C units, the number of weights is reduced\n    from MxN to (M+N)xC. The weights are obtained by fitting a neural network to\n    predict the same output as the original model.\n    '

    def __init__(self, **kwargs):
        (super(InsertDenseSparse, self).__init__)(**kwargs)
        self.target_layer_type = 'dense'

    def compress_layer(self, layer_name, units=16, iterations=100, verbose=False):
        self.logger.info('Searching for layer: {}'.format(layer_name))
        self.layer_name = layer_name
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        weights, bias = layer.get_weights()
        weights = tf.Variable(weights, dtype='float32')
        units = units
        w_init = tf.random_normal_initializer()
        basis = tf.Variable(name='basis', initial_value=w_init(shape=(weights.shape[0], units)))
        sparse_dict = tf.Variable(name='sparse_code', initial_value=np.random.randint(0,
          2, size=(units, weights.shape[(-1)])),
          constraint=(BinaryWeight(units)),
          dtype='float32')
        optimizer = tf.keras.optimizers.Adam()
        for i in range(iterations):
            with tf.GradientTape() as (tape):
                pred = tf.matmul(basis, sparse_dict)
                loss = tf.reduce_sum(tf.nn.l2_loss(weights - pred))
            gradients = tape.gradient(loss, [basis, sparse_dict])
            optimizer.apply_gradients(zip(gradients, [basis, sparse_dict]))
            if verbose and i % 10 == 0:
                self.logger.debug('Epoch {} Loss: {}'.format(i, loss))

        self.logger.info('Creating layer with new weights.')
        inputs = tf.keras.layers.Input(shape=(self.input_shape))
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            if layer.name == layer_name:
                new_layer = DenseSparseSVD(units0=units, units1=(weights.shape[(-1)]), weights0=(basis.numpy()), weights1=(sparse_dict.numpy()), bias1=bias,
                  activation=(layer.get_config()['activation']),
                  name=(layer_name + '/SparseSVD'))
                new_layer._name = layer_name + '/SparseSVD'
                x = new_layer(x)
            else:
                layer.trainable = False
                x = layer(x)

        self.model = tf.keras.Model(inputs, x)
        self.model.compile(optimizer=(self.optimizer), loss=(self.loss_object),
          metrics=(self.metrics))
        self.logger.info('Fine-tuning layer.')
        self.model.fit((self.dataset), epochs=(self.tuning_epochs), verbose=(self.fit_verbose))
        for layer in self.model.layers:
            layer.trainable = True

        self.new_layer_name = layer_name + '/SparseSVD'
        idx_sparse_layer = self.find_layer(layer_name + '/SparseSVD')
        layer = self.model.layers[idx_sparse_layer]
        sparse_weights = layer.layers[(-1)].get_weights()[0]
        basis_weights = layer.layers[0].get_weights()[0]
        num_zeroes_sparse = tf.math.count_nonzero(sparse_weights == 0.0).numpy()
        non_zeroes_sparse = tf.math.count_nonzero(sparse_weights != 0.0).numpy()
        self.logger.info('Sparse dict has {} zeroes before fine-tuning.'.format(num_zeroes_sparse))
        num_zeroes_basis = tf.math.count_nonzero(basis_weights == 0.0).numpy()
        self.logger.info('Basis dict has {} zeroes before fine-tuning.'.format(num_zeroes_basis))
        if self.fine_tune:
            self.fine_tune_model()
        layer = self.model.layers[idx_sparse_layer]
        sparse_weights = layer.layers[(-1)].get_weights()[0]
        basis_weights = layer.layers[0].get_weights()[0]
        num_zeroes_sparse = tf.math.count_nonzero(sparse_weights == 0.0).numpy()
        non_zeroes_sparse = tf.math.count_nonzero(sparse_weights != 0.0).numpy()
        self.logger.info('Sparse dict has {} zeroes after fine-tuning.'.format(num_zeroes_sparse))
        num_zeroes_basis = tf.math.count_nonzero(basis_weights == 0.0).numpy()
        self.logger.info('Basis dict has {} zeroes after fine-tuning.'.format(num_zeroes_basis))
        self.weights_after = self.weights_before - (tf.size(weights) - (non_zeroes_sparse + tf.size(basis)))
        self.logger.info('Finished compression')


class ConvSVD(tf.keras.Model):

    def __init__(self, units, kernel_size, filters, name, **kwargs):
        (super(ConvSVD, self).__init__)(name, **kwargs)
        self.conv0 = tf.keras.layers.Conv2D(units, kernel_size, activation='relu', name=(name + '/conv_0'), padding='same')
        self.conv1 = tf.keras.layers.Conv2D(filters,
          kernel_size, activation='relu', name=(name + '/conv_1'))

    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        return x


class InsertSVDConv(ModelCompression):
    __doc__ = '\n    Compression techniques that reduces the number of weights in a filter by\n    applying the filter in two steps, one horizontal and one vertical. Instead of\n    using a DxD filter, a Dx1 is used followed by a 1xD filter. Thus, reducing the\n    number of required weights. The process to find the weights of the one\n    dimensional filters is by regressing a convolutional neural network and\n    setting the output of the original filter as the target of the regression\n    model.\n    '

    def __init__(self, **kwargs):
        (super(InsertSVDConv, self).__init__)(**kwargs)
        self.target_layer_type = 'conv'

    def compress_layer(self, layer_name, units=32, iterations=5):
        self.logger.info('Searching for layer: {}'.format(layer_name))
        self.layer_name = layer_name
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        filters = layer.get_config()['filters']
        kernel_size = layer.get_config()['kernel_size']
        inputs = tf.keras.layers.Input(shape=(self.input_shape))
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            if layer.name == layer_name:
                svd_conv = ConvSVD(units=units, filters=filters, kernel_size=kernel_size,
                  name=(layer_name + '/SVDConv'))
                svd_conv._name = layer_name + '/SVDConv'
                removed_weights = np.sum([K.count_params(w) for w in layer.trainable_weights])
                x = svd_conv(x)
            else:
                layer.trainable = False
                x = layer(x)

        self.model = tf.keras.Model(inputs, x)
        self.model.compile(optimizer=(self.optimizer), loss=(self.loss_object),
          metrics=(self.metrics))
        self.model.fit((self.dataset), epochs=(self.tuning_epochs), verbose=(self.fit_verbose))
        for layer in self.model.layers:
            layer.trainable = True

        self.new_layer_name = layer_name + '/SVDConv'
        added_weights = np.sum([K.count_params(w) for w in svd_conv.trainable_weights])
        if self.fine_tune:
            self.fine_tune_model()
        self.weights_after = self.weights_before - removed_weights + added_weights
        self.logger.info('Finished compression')


class DepthwiseSeparableConvolution(ModelCompression):
    __doc__ = '\n    Compression techniques that reduces the number of weights in a filter by\n    applying the filter in two steps, one horizontal and one vertical. Instead of\n    using a DxD filter, a Dx1 is used followed by a 1xD filter. Thus, reducing the\n    number of required weights. The process to find the weights of the one\n    dimensional filters is by regressing a convolutional neural network and\n    setting the output of the original filter as the target of the regression\n    model.\n    '

    def __init__(self, **kwargs):
        (super(DepthwiseSeparableConvolution, self).__init__)(**kwargs)
        self.target_layer_type = 'conv'

    def compress_layer(self, layer_name, iterations=5):
        self.weights_before = self.count_trainable_weights()
        self.logger.info('Searching for layer: {}'.format(layer_name))
        self.layer_name = layer_name
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        filters = layer.get_config()['filters']
        kernel_size = layer.get_config()['kernel_size']
        inputs = tf.keras.layers.Input(shape=(self.input_shape))
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            if layer.name == layer_name:
                new_layer = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=kernel_size,
                  name=(layer_name + '/DepthwiseSeparableLayer'),
                  trainable=True)
                x = new_layer(x)
                removed_weights = np.sum([K.count_params(w) for w in layer.trainable_weights])
                added_weights = np.sum([K.count_params(w) for w in new_layer.trainable_weights])
            else:
                layer.trainable = False
                x = layer(x)

        self.model = tf.keras.Model(inputs, x)
        self.model.compile(optimizer=(self.optimizer), loss=(self.loss_object),
          metrics=(self.metrics))
        self.model.fit((self.dataset), epochs=(self.tuning_epochs), verbose=(self.fit_verbose))
        for layer in self.model.layers:
            layer.trainable = True

        self.new_layer_name = layer_name + '/DepthwiseSeparableLayer'
        self.logger.info('Replaced layer was using {} weights.'.format(removed_weights))
        self.logger.info('DepthWise Separable is using {} weights.'.format(added_weights))
        if self.fine_tune:
            self.fine_tune_model()
        self.weights_after = self.weights_before - removed_weights + added_weights
        self.logger.info('Finished compression')


class FireLayer(tf.keras.layers.Layer):

    def __init__(self, s1x1, e1x1, e3x3, name, kernel_size=(3,3), strides=1, padding='same',**kwargs):
        (super(FireLayer, self).__init__)(name, **kwargs)
        self.squeeze = tf.keras.layers.Conv2D(s1x1,
          (1, 1), activation='relu', name=(name + '/squeeze'))
        self.expand1x1 = tf.keras.layers.Conv2D(e1x1,
          (1, 1), padding='valid', name=(name + '/expand1x1'))
        self.expand3x3 = tf.keras.layers.Conv2D(e3x3,
          kernel_size, strides=strides, padding=padding, name=(name + '/expand3x3'))
        self.padding = padding

    def get_config(self):
        config1x1 = self.expand1x1.get_config()
        config = self.expand3x3.get_config().copy()
        config.update({'filters': config['filters'] + config1x1['filters']})
        return config

    def call(self, inputs):
        x = self.squeeze(inputs)
        o1x1 = self.expand1x1(x)
        o3x3 = self.expand3x3(x)
        x = tf.keras.layers.concatenate([o1x1, o3x3], axis=3)
        if self.padding == 'valid':
            x = tf.keras.layers.Cropping2D(cropping=1)(x)
        return x


class FireLayerCompression(ModelCompression):
    __doc__ = '\n    Compression techniques that replaces a convolutional layer by a fire layer,\n    which consists of 1x1 and 3x3 convolutions. A 1x1 is\n    '

    def __init__(self, **kwargs):
        (super(FireLayerCompression, self).__init__)(**kwargs)
        self.target_layer_type = 'conv'

    def compress_layer(self, layer_name, iterations=2):
        self.logger.info('Searching for layer: {}'.format(layer_name))
        self.layer_name = layer_name
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]

        filters = layer.get_config()['filters']
        kernel_size = layer.get_config()['kernel_size']
        activation = layer.get_config()['activation']
        strides = layer.get_config()['strides']
        padding = layer.get_config()['padding']

        inputs = tf.keras.layers.Input(shape=(self.input_shape))

        fire_layer = tf.keras.Sequential([
         FireLayer(s1x1=(filters // 4),
                  e1x1=(filters // 2),
                  e3x3=(filters // 2),
                  padding=padding,
                  strides=strides,
                  name=(layer_name + '/FireLayer'))])
        x = self.model.input
        output = self.model.layers[(idx - 1)].output
        generate_input = tf.keras.Model(x, output)
        output = self.model.layers[idx].output
        generate_output = tf.keras.Model(x, output)
        fire_layer.compile(optimizer='adam', loss='mae')
        self.logger.info('Learning FireLayer filter.')

        def generate_new_dataset(image, label):
            return generate_input(image), generate_output(image)
 
        dataset = self.dataset.map(generate_new_dataset)

        fire_layer.fit(dataset, epochs=iterations, verbose=1)

        inputs = tf.keras.layers.Input(shape=(self.input_shape))
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            if layer.name == layer_name:
                new_layer = FireLayer(s1x1=(filters // 4),
                  e1x1=(filters // 2),
                  e3x3=(filters // 2),
                  padding=padding,
                  strides=strides,
                  name=layer_name + '/FireLayer')
                _ = new_layer(x)
                new_layer.set_weights(fire_layer.layers[0].get_weights())
                x = new_layer(x)
                
                removed_weights = np.sum([K.count_params(w) for w in layer.trainable_weights])
                added_weights = np.sum([K.count_params(w) for w in new_layer.trainable_weights])
            else:
                x = layer(x)

        self.model = tf.keras.Model(inputs, x)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_object, metrics=self.metrics)

        self.new_layer_name = layer_name + '/FireLayer'
        self.logger.info('Replaced layer was using {} weights.'.format(removed_weights))
        self.logger.info('Firelayer is using {} weights.'.format(added_weights))
        if self.fine_tune:
            self.fine_tune_model()
        self.weights_after = self.weights_before - removed_weights + added_weights
        self.logger.info('Finished compression')

class MLPConv(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides=1, padding='VALID', activation='relu', *args, **kwargs):
        (super(MLPConv, self).__init__)(*args, **kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        if isinstance(strides, int):
            self.strides = (
             strides, strides)
        else:
            if isinstance(strides, tuple) or isinstance(strides, list):
                self.strides = strides
        self.activation = tf.keras.activations.get(activation)
        self.padding = padding.upper()

    def build(self, input_shape):
        batch, h, w, channels = input_shape
        w_init = tf.random_normal_initializer()
        self.w_0 = tf.Variable(name='kernel0', initial_value=w_init(shape=(tf.reduce_prod(self.kernel_size) * channels, self.filters // 2), dtype='float32'),
          trainable=True)
        self.w_1 = tf.Variable(name='kernel1', initial_value=w_init(shape=(self.filters // 2, self.filters), dtype='float32'),
          trainable=True)
        b_init = tf.zeros_initializer()
        self.b_0 = tf.Variable(name='bias0', initial_value=b_init(shape=(
         self.filters // 2,),
          dtype='float32'),
          trainable=True)
        self.b_1 = tf.Variable(name='bias1', initial_value=b_init(shape=(
         self.filters,),
          dtype='float32'),
          trainable=True)
        fh = self.kernel_size[0]
        fw = self.kernel_size[1]
        self.indexes = tf.constant([[x, y] for x in range(fh // 2, h - 1) for y in range(fw // 2, w - 1)])
 

    def get_config(self):
        config = super().get_config().copy()
        config.update({'filters':self.filters, 
         'kernel_size':self.kernel_size})
        return config

    @tf.function
    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]

        fh, fw = self.kernel_size
        stride_v, stride_h = self.strides
        patches = tf.image.extract_patches(images=inputs, sizes=[
         1, fh, fw, 1],
          strides=[
         1, stride_v, stride_h, 1],
          rates=[
         1, 1, 1, 1],
          padding=self.padding)
        output = self.activation(tf.matmul(self.activation(tf.matmul(patches, self.w_0) + self.b_0), self.w_1) + self.b_1)
        return output


class MLPCompression(ModelCompression):
    __doc__ = '\n    Compression techniques that replaces a convolutional layer by a Multi-layer\n    perceptron that learns to generate the output of each filter.\n    '

    def __init__(self, **kwargs):
        (super(MLPCompression, self).__init__)(**kwargs)
        self.target_layer_type = 'conv'

    def compress_layer(self, layer_name, iterations=2, batch_size=32):
        self.logger.info('Searching for layer: {}'.format(layer_name))
        self.layer_name = layer_name
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        filters = layer.get_config()['filters']
        kernel_size = layer.get_config()['kernel_size']
        activation = layer.get_config()['activation']
        padding = layer.get_config()['padding']
        mlp_model = tf.keras.Sequential([
         MLPConv(kernel_size=kernel_size, filters=filters,
           activation=activation, padding=padding,
           name=(layer_name + '/MLPConv'))])
        x = self.model.input
        output = self.model.layers[(idx - 1)].output
        generate_input = tf.keras.Model(x, output)
        output = self.model.layers[idx].output
        generate_output = tf.keras.Model(x, output)
        mlp_model.compile(optimizer='adam', loss='mae')
        self.logger.info('Learning MLP filter.')

        def generate_new_dataset(image, label):
            return generate_input(image), generate_output(image)
 
        dataset = self.dataset.map(generate_new_dataset)

        mlp_model.fit(dataset, epochs=iterations, verbose=1)

        inputs = tf.keras.layers.Input(shape=(self.input_shape))
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            if layer.name == layer_name:
                new_mlp = MLPConv(kernel_size=kernel_size, filters=filters, padding=padding,
                  activation=activation,
                  name=(layer_name + '/MLPConv'))
                _ = new_mlp(x)
                new_mlp.set_weights(mlp_model.layers[0].get_weights())
                x = new_mlp(x)
                
                removed_weights = np.sum([K.count_params(w) for w in layer.trainable_weights])
                added_weights = np.sum([K.count_params(w) for w in new_mlp.trainable_weights])
            else:
                x = layer(x)

        self.model = tf.keras.Model(inputs, x)
        self.model.compile(optimizer=(self.optimizer), loss=(self.loss_object),
          metrics=(self.metrics))

        self.new_layer_name = layer_name + '/MLPConv'
        if self.fine_tune:
            self.fine_tune_model()
        self.logger.info('Replaced layer was using {} weights.'.format(removed_weights))
        self.logger.info('MLPConv is using {} weights.'.format(added_weights))
        self.weights_after = self.weights_before - removed_weights + added_weights
        self.logger.info('Finished compression')


class SparseConnectionConv2D(tf.keras.layers.Conv2D):

    def __init__(self, sparse_connections, *args, **kwargs):
        (super(SparseConnectionConv2D, self).__init__)(*args, **kwargs)
        self.sparse_connections = sparse_connections
        self.zero_init = tf.zeros_initializer()

    def convolution_op(self, inputs, kernel):
        """
        Exact copy from the method of Conv class.
        """
        if self.padding == 'causal':
            tf_padding = 'VALID'
        else:
            if isinstance(self.padding, str):
                tf_padding = self.padding.upper()
            else:
                tf_padding = self.padding
        return tf.nn.convolution(inputs,
          kernel,
          strides=(list(self.strides)),
          padding=tf_padding,
          dilations=(list(self.dilation_rate)),
          data_format=(self._tf_data_format),
          name=(self.__class__.__name__))

    def build(self, input_shape):
        super().build(input_shape)
        batch, h, w, channels = input_shape
        self.conn_tensor = tf.constant(self.sparse_connections)
        self.conn_tensor = tf.reshape((self.conn_tensor), shape=[
         self.filters, channels])
        self.indexes = tf.constant([[x, y] for x in range(self.filters) for y in range(channels)])
        zeroes_init = tf.zeros_initializer()
        self.sparse_kernel = tf.Variable(zeroes_init(shape=(self.kernel.shape), dtype=(tf.float32)), name='sparse_kernel',
          trainable=False)
        for filter in tf.range(self.filters):
            for channel in tf.range(self.kernel.shape[(-2)]):
                if self.conn_tensor[(filter, channel)] == 1:
                    self.sparse_kernel[:, :, channel, filter].assign(self.kernel[:, :, channel, filter])

    def set_connections(self, connections):
        self.sparse_connections = connections
        self.conn_tensor = tf.constant(self.sparse_connections)
        self.conn_tensor = tf.reshape((self.conn_tensor), shape=[
         self.filters, self.kernel.shape[(-2)]])
        for filter in tf.range(self.filters):
            for channel in tf.range(self.kernel.shape[(-2)]):
                if self.conn_tensor[(filter, channel)] == 1:
                    self.sparse_kernel[:, :, channel, filter].assign(self.kernel[:, :, channel, filter])

    def get_connections(self):
        return self.sparse_connections

    @tf.function
    def call(self, inputs):
        return self.convolution_op(inputs, self.sparse_kernel)


class AddSparseConnectionsCallback(tf.keras.callbacks.Callback):

    def __init__(self, layer_idx, target_perc=0.75, conn_perc_per_epoch=0.1):
        super(AddSparseConnectionsCallback, self).__init__()
        self.target_perc = target_perc
        self.conn_perc_per_epoch = conn_perc_per_epoch
        self.layer_idx = layer_idx
        self.logger = logging.getLogger(__name__)

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info('Updating connections sparse connections.')
        connections = np.asarray(self.model.layers[self.layer_idx].get_connections())
        total_connections = connections.shape[0]
        num_connections = np.sum(connections)
        perc_connections = num_connections / total_connections
        if perc_connections < self.target_perc:
            select_n = int(total_connections * self.conn_perc_per_epoch)
            remaining_to_goal = int(total_connections * (self.target_perc - perc_connections))
            smallest = min(select_n, remaining_to_goal)
            missing_connections = np.squeeze(np.argwhere(connections == 0))
            self.logger.info('Adding {} connections of {} missing.'.format(smallest, missing_connections.shape[0]))
            new_connections = np.random.choice(missing_connections, smallest,
              replace=False)
            connections[new_connections] = 1
            self.logger.info('Number of activated connections {} of {}.'.format(np.sum(connections), connections.shape[0]))
            self.model.layers[self.layer_idx].set_connections(connections.tolist())


class SparseConnectionsCompression(ModelCompression):
    __doc__ = '\n    Compression technique that sparsifies the connections between input channels\n    and output channels when applying filters in a convolutional layer.\n    '

    def __init__(self, **kwargs):
        (super(SparseConnectionsCompression, self).__init__)(**kwargs)
        self.target_layer_type = 'conv'

    def compress_layer(self, layer_name, epochs=20, target_perc=0.75, conn_perc_per_epoch=0.1):
        self.logger.info('Searching for layer: {}'.format(layer_name))
        self.layer_name = layer_name
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        filters = layer.get_config()['filters']
        kernel_size = layer.get_config()['kernel_size']
        activation = layer.get_config()['activation']
        input_channels = self.model.layers[(idx - 1)].output.shape[(-1)]
        self.logger.info('Creating model with Sparse Connections Layer.')
        total_connections = filters * input_channels
        connections = np.zeros(total_connections, dtype=(np.uint8))
        remaining_connections = list(range(total_connections))
        new_connections = np.random.choice(remaining_connections, (math.ceil(connections.shape[0] / 100)),
          replace=False)
        connections[new_connections] = 1
        for conn in new_connections:
            remaining_connections.remove(conn)

        inputs = tf.keras.layers.Input(shape=(self.input_shape))
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            if layer.name == layer_name:
                layer_weights = layer.get_weights()
                layer_weights.append(layer_weights[0])
                sparse_c_layer = SparseConnectionConv2D(sparse_connections=connections, filters=filters,
                  activation=activation,
                  kernel_size=kernel_size,
                  name=(layer_name + '/SparseConnectionsConv'))
                _ = sparse_c_layer(x)
                sparse_c_layer.set_weights(layer_weights)
                x = sparse_c_layer(x)
                removed_weights = np.sum([K.count_params(w) for w in layer.trainable_weights])
                added_weights = np.sum([K.count_params(w) for w in sparse_c_layer.trainable_weights])
            else:
                x = layer(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        self.model.compile(optimizer=(self.optimizer), loss=(self.loss_object),
          metrics=(self.metrics))
        layer_idx = self.find_layer(layer_name + '/SparseConnectionsConv')
        cb = AddSparseConnectionsCallback(layer_idx, target_perc=target_perc,
          conn_perc_per_epoch=conn_perc_per_epoch)
        self.model.fit((self.dataset), epochs=epochs, callbacks=[cb], verbose=(self.fit_verbose))
        self.new_layer_name = layer_name + '/SparseConnectionsConv'
        connections = self.model.layers[layer_idx].get_connections()
        num_zeroes = len(connections) - np.sum(connections)
        self.logger.info('Replaced layer was using {} weights.'.format(removed_weights))
        self.logger.info('Sparse Connections is using {} weights. It has {} connections and {} zeros. '.format(added_weights, connections, num_zeroes))
        self.weights_after = self.weights_before - num_zeroes
        self.logger.info('Compressed model has {} connections.'.format(np.sum(connections)))
        if self.fine_tune:
            self.fine_tune_model()
        self.logger.info('Finished compression')


class L1L2SRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, x):
        regularization = tf.constant(0.0, dtype=(tf.float32))
        if self.l1:
            regularization += self.l1 * tf.reduce_sum(tf.abs(x))
        if self.l2:
            regularization += self.l2 * tf.reduce_sum(tf.square(x))
        return regularization

    def get_config(self):
        return {'l1':self.l1, 
         'l2':self.l2}


class SInitializer(tf.keras.initializers.Initializer):

    def __init__(self, S):
        self.s = S

    def __call__(self, shape, dtype=None, **kwargs):
        return self.s


class SparseConvolution2D(tf.keras.layers.Layer):

    def __init__(self, kernel_size, filters, PQS, bases, padding='valid', strides=1, activation=None, *args, **kwargs):
        (super(SparseConvolution2D, self).__init__)(*args, **kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.PQS = PQS
        self.activation = tf.keras.activations.get(activation)
        self.bases = bases
        self.padding = padding.upper()
        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            if isinstance(strides, tuple) or isinstance(strides, list):
                self.strides = strides

    def build(self, input_shape):
        batch, h, w, channels = input_shape
        sh, sw = self.kernel_size
        w_init = tf.random_normal_initializer()
        identity_initializer = tf.initializers.Identity()
        self.S = self.add_weight(name='S', initializer=(SInitializer(self.PQS[(-1)])),
          shape=(
         channels, self.bases, self.filters),
          dtype='float32',
          trainable=True,
          regularizer=L1L2SRegularizer(l1=0.5, l2=0.5))
        zeroes = tf.zeros_initializer()
        self.Q = tf.Variable(name='Q', initial_value=zeroes(shape=(
         channels, sh, sw, self.bases),
          dtype='float32'),
          trainable=True)
        self.Q.assign(self.PQS[1])
        self.P = tf.Variable(name='P', initial_value=identity_initializer(shape=(channels, channels), dtype='float32'),
          trainable=True)
        self.P.assign(self.PQS[0])
        del self.PQS
        super().build(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'filters':self.filters, 
         'kernel_size':self.kernel_size})
        return config

    # @tf.function
    def call(self, inputs):
        print('start')
        batch, h, w, channels = inputs.shape

        padding = self.padding
        strides = self.strides

        def calculate_J(I, P, channel):
            temp_P = tf.slice(P, begin=[channel, 0], size=[1, -1])
            temp_P = tf.reshape(temp_P, shape=[-1])
            result = tf.reduce_sum((tf.multiply(I, temp_P)), axis=(-1))
            return result

        def func1(i):
            return calculate_J(inputs, self.P, i)

        print('calculate J')
        J = tf.vectorized_map(func1, elems=tf.range(channels))

        def calculate_tau(J, Q, channel):
            Ji = tf.slice(J, begin=[channel, 0, 0, 0], size=[1, -1, -1, -1])
            Ji = tf.transpose(Ji, perm=[1, 2, 3, 0])
            Qi = tf.slice(Q, begin=[channel, 0, 0, 0], size=[1, -1, -1, -1])
            Qi = tf.transpose(Qi, perm=[1, 2, 0, 3])
            taui = tf.nn.conv2d(input=Ji, filters=Qi, strides=1,
              padding='SAME')
            return taui

        def func2(i):
            return calculate_tau(J, self.Q, i)

        print('calculate T')
        tau = tf.map_fn(func2, elems=tf.range(channels), fn_output_signature=tf.float32)
        tau = tf.transpose(tau, perm=[1, 2, 3, 4, 0])
        filters = self.S.shape[(-1)]

        def calculate_O(T, S, filter_id):
            S = tf.slice(S, begin=[0, 0, filter_id], size=[-1, -1, 1])
            S = tf.squeeze(S)
            batch, h, w, bases, channels = T.shape
            batch = tf.shape(T)[0]
            S = tf.reshape(S, shape=[bases * channels])
            T = tf.reshape(T, shape=[batch, h, w, bases * channels])
            O = tf.reduce_sum((tf.multiply(T, S)), axis=(-1))
            return O

        def func3(i):
            return calculate_O(tau, self.S, i)

        print('calculate O')
        O = tf.vectorized_map(func3, elems=(tf.range(filters)))
        O = tf.transpose(O, perm=[1, 2, 3, 0])
        return O


class SparseConvolutionCompression(ModelCompression):
    __doc__ = '\n    Compression technique that performs an sparse convolution\n    '

    def __init__(self, **kwargs):
        (super(SparseConvolutionCompression, self).__init__)(**kwargs)
        self.bases = None
        self.target_layer_type = 'conv'

    def find_pqs(self, layer, iterations=50, verbose=True):
        w_init = tf.random_normal_initializer()
        identity_initializer = tf.initializers.Identity()
        weights = tf.constant(layer.get_weights()[0])
        sh, sw, channels, filters = weights.shape
        R = tf.Variable(name='R', initial_value=w_init(shape=(weights.shape), dtype='float32'),
          trainable=False)
        P = tf.Variable(name='P', initial_value=identity_initializer(shape=(channels, channels), dtype='float32'),
          trainable=True)
        self.logger.info('Searching for matrix P.')
        indexes = tf.constant([[i, j] for i in range(channels) for j in range(filters)])

        def calculate_kernel(R, P, channel, filter_id):
            R = tf.slice(R, begin=[0, 0, 0, filter_id], size=[-1, -1, -1, 1])
            P = tf.slice(P, begin=[0, channel], size=[-1, 1])
            R = tf.squeeze(R)
            P = tf.squeeze(P)
            return tf.reduce_sum((R * P), axis=(-1))

        def func(idx):
            return calculate_kernel(R, P, idx[0], idx[1])

        optimizer = tf.keras.optimizers.Adam()
        for i in range(iterations):
            with tf.GradientTape() as (tape):
                output = tf.vectorized_map(func, elems=indexes)
                pred = tf.reshape(output, shape=[sh, sw, channels, filters])
                loss = tf.reduce_mean(tf.nn.l2_loss(weights - pred))
            gradients = tape.gradient(loss, [P])
            optimizer.apply_gradients(zip(gradients, [P]))
            if verbose and i % 10 == 0:
                self.logger.info('Epoch {} of RxP Loss: {}'.format(i, loss))

        self.logger.info('Searching for matrices Q and S.')
        zeroes = tf.zeros_initializer()
        S = tf.Variable(name='S', initial_value=zeroes(shape=(
         channels, self.bases, filters),
          dtype='float32'),
          trainable=True)
        Q = tf.Variable(name='Q', initial_value=zeroes(shape=(
         channels, sh, sw, self.bases),
          dtype='float32'),
          trainable=True)
        for i in range(sh):
            Q[:, i, i, :].assign(tf.ones(shape=(channels, self.bases)))

        def calculate_SQ_output(S, Q, channel, filter_id):
            S = tf.slice(S, begin=[channel, 0, filter_id], size=[1, -1, 1])
            Q = tf.slice(Q, begin=[channel, 0, 0, 0], size=[1, -1, -1, -1])
            S = tf.squeeze(S)
            Q = tf.squeeze(Q)
            return tf.reduce_sum((Q * S), axis=(-1))

        indexes = tf.constant([[i, j] for i in range(channels) for j in range(filters)])

        def func(idx):
            return calculate_SQ_output(S=S,
              Q=Q,
              channel=(idx[0]),
              filter_id=(idx[1]))

        optimizer = tf.keras.optimizers.Adam()
        for i in range(iterations):
            with tf.GradientTape() as (tape):
                output = tf.vectorized_map(func, elems=indexes)
                pred = tf.reshape(output, shape=[sh, sw, channels, filters])
                loss = tf.reduce_mean(tf.nn.l2_loss(R - pred))
            gradients = tape.gradient(loss, [S, Q])
            optimizer.apply_gradients(zip(gradients, [S, Q]))
            if verbose and i % 10 == 0:
                self.logger.info('Epoch {} Loss: {}'.format(i, loss))

        self.logger.info('Matrix P has shape {}'.format(P.shape))
        self.logger.info('Matrix Q has shape {}'.format(Q.shape))
        self.logger.info('Matrix S has shape {}'.format(S.shape))
        return (
         P, Q, S)

    def compress_layer(self, layer_name, bases, iterations=5):
        self.bases = bases
        self.weights_before = self.count_trainable_weights()
        self.logger.info('Searching for layer: {}'.format(layer_name))
        self.layer_name = layer_name
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        filters = layer.get_config()['filters']
        activation = layer.get_config()['activation']
        kernel_size = layer.get_config()['kernel_size']
        padding = layer.get_config()['padding']
        P, Q, S = self.find_pqs(layer, iterations=iterations)
        inputs = tf.keras.layers.Input(shape=(self.input_shape))
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            if layer.name == layer_name:
                new_layer = SparseConvolution2D(kernel_size=kernel_size, filters=filters, padding=padding,
                  PQS=[
                 P.numpy(), Q.numpy(),
                 S.numpy()],
                  activation=activation,
                  bases=(self.bases),
                  name=(layer_name + '/SparseConv2D'))
                x = new_layer(x)
                removed_weights = np.sum([K.count_params(w) for w in layer.trainable_weights])
            else:
                layer.trainable = False
                x = layer(x)

        self.model = tf.keras.Model(inputs, x)
        self.model.compile(optimizer=(self.optimizer), loss=(self.loss_object),
          metrics=(self.metrics))
        self.model.fit((self.dataset), epochs=(self.tuning_epochs), verbose=(self.fit_verbose))
        for layer in self.model.layers:
            layer.trainable = True

        self.new_layer_name = layer_name + '/SparseConv2D'
        num_zeroes_sparse = tf.math.count_nonzero(S == 0.0).numpy()
        added_weights = tf.reduce_prod(P.shape) + tf.reduce_prod(Q.shape) + (tf.reduce_prod(S.shape) - num_zeroes_sparse)
        self.logger.info('Replaced layer was using {} weights.'.format(removed_weights))
        self.logger.info('Sparse Conv2D is using {} weights. It has {} zeroes that were not counted. '.format(added_weights, num_zeroes_sparse))
        if self.fine_tune:
            self.fine_tune_model()
        self.weights_after = self.weights_before
        self.logger.info('Finished compression')
# okay decompiling CompressionTechniques.cpython-36.pyc
