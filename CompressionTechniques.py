import tensorflow as tf
import numpy as np
import logging
import copy
import random
import re
from deap import base, creator, tools, algorithms


class ModifiableModel(tf.keras.Model):
    def __init__(self, layers):
        super(ModifiableModel, self).__init__()
        self.model_layers = layers

    def __call__(self, inputs):
        x = self.model_layers[0](inputs)

        for layer in self.model_layers[1:]:
            x = layer(x)

        return x


class ModelCompression:
    """
    Base class for compressing a deep learning model. The class takes a tensorflow
    model and a dataset that will be used to fit a regression.
    """

    def __init__(self, model, optimizer, loss, metrics, dataset=None, fine_tune=True, epochs=10):
        """

        :param model: tensorflow model that will be optimized.
        :param optimizer: optimizer that will be used to compile the optimized model.
        :param loss: loss object that will be used to compile the optimized model.
        :param metrics: metrics that will be used to compile the optimized model.
        :param dataset: dataset that will be used for compressing a layer and/or fine-tuning.
        :param fine_tune: flag to select if the optimized model will be trained.
        :param epochs: number of epochs of the fine-tuning.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss
        self.metrics = metrics
        self.dataset = dataset
        self.fine_tune = fine_tune
        self.epochs = epochs
        self.logger = logging.getLogger(__name__)
        self.model_changes = {}

    def get_technique(self):
        return self.__class__.__name__

    def find_layer(self, layer_name: str) -> int:
        names = [layer.name for layer in self.model.layers]
        return names.index(layer_name)

    def update_model(self):
        self.logger.info('Updating the model.')
        layers = self.model.layers
        for idx, value in reversed(list(self.model_changes.items())):
            if value['action'] == 'delete':
                layers.pop(idx)
            elif value['action'] == 'insert':
                layers.insert(idx, value['layer'])
            elif value['action'] == 'replace':
                layers[idx] = value['layer']

        try:
            input = layers[0].input
            x = layers[0](input)
            for layer in layers[1:]:
                x = layer(x)
            self.model = tf.keras.Model(inputs=input, outputs=x)
            self.model.compile(optimizer=self.optimizer, loss=self.loss_object, metrics=self.metrics)
            self.logger.info('Model updated.')
        except ValueError:
            self.logger.error('The input and the weights of a layer do not match.')
            raise

    def get_model(self):
        return self.model

    def compress_layer(self, layer_name: str):
        pass


class DeepCompression(ModelCompression):
    """
    Compression technique that sets to 0 all weights that are below a threshold in
    a Dense layer. No clustering and Huffman encoding is performed.
    """

    def __init__(self, threshold: float = 0.0001, **kwargs):
        super(DeepCompression, self).__init__(**kwargs)
        self.threshold = threshold

    def compress_layer(self, layer_name: str):
        self.logger.info('Searching for layer: {}'.format(layer_name))
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        weights = layer.get_weights()
        tf_weights = tf.Variable(weights[0], trainable=True)
        below_threshold = tf.abs(tf_weights) < self.threshold
        self.logger.info('Pruned weights: {}'.format(tf.math.count_nonzero(below_threshold)))
        new_weights = tf.where(below_threshold, 0.0, tf_weights)
        layer.set_weights([new_weights, weights[1]])
        self.logger.debug('Found {} weights with value 0.0'.format(tf.math.count_nonzero(
            tf.abs(self.model.layers[idx].get_weights()[0]) == 0.0)))
        self.logger.info('Finished compression')


class ReplaceDenseWithGlobalAvgPool(ModelCompression):
    """
    Compression technique that replaces all dense and flatten layers
    between the last convolutional layer and the softmax layer with a GlobalAveragePooling2D layer.
    """

    def __init__(self, **kwargs):
        super(ReplaceDenseWithGlobalAvgPool, self).__init__(**kwargs)

    def compress_layer(self):
        self.logger.info('Searching for all dense layers.')
        regexp = re.compile(r'dense|fc|flatten')
        for idx, layer in enumerate(self.model.layers):
            config = layer.get_config()
            lname = config['name'].lower()

            if regexp.search(lname):
                if 'dense' in lname and config['activation'] == 'softmax':
                    continue
                self.model_changes[idx] = {'action': 'delete',
                                           'layer': None}

        self.model_changes[len(self.model.layers) - 2] = {'action': 'replace',
                                                          'layer': tf.keras.layers.GlobalAveragePooling2D()}
        self.logger.info('Number of changes required: {}.'.format(len(self.model_changes)))
        self.update_model()
        if self.fine_tune:
            self.logger.info('Fine-tuning the optimized model.')
            self.model.fit(self.dataset, epochs=self.epochs)
        self.logger.info('Finished compression')


class InsertDenseSVD(ModelCompression):
    """
    Compression technique that inserts a smaller dense layer using Singular Value Decomposition. By
    inserting a smaller layer with C units, the number of weights is reduced
    from MxN to (M+N)xC. The weights are obtained by fitting a neural network to
    predict the same output as the original model.
    """

    def __init__(self, **kwargs):
        super(InsertDenseSVD, self).__init__(**kwargs)

    def compress_layer(self, layer_name, iterations=1000, verbose=False):
        self.logger.info('Searching for layer: {}'.format(layer_name))

        # Find layer
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        weights, bias = layer.get_weights()
        units = weights.shape[0] // 6
        weights = tf.Variable(weights, dtype='float32')

        # Create new weights for the two dense layers.
        w_init = tf.random_normal_initializer()
        w1 = tf.Variable(name='w1',
                         initial_value=w_init(shape=(weights.shape[0], units)))

        w2 = tf.Variable(name='w2',
                         initial_value=w_init(shape=(units, weights.shape[1])))

        # Learn new weights.
        optimizer = tf.keras.optimizers.Adam()
        for i in range(iterations):
            with tf.GradientTape() as tape:
                pred = tf.matmul(w1, w2)
                loss = tf.reduce_sum(tf.nn.l2_loss(weights - pred))

            gradients = tape.gradient(loss, [w1, w2])
            optimizer.apply_gradients(zip(gradients, [w1, w2]))
            if verbose and i % 10000 == 0:
                print('Epoch {} Loss: {}'.format(i, loss))

        self.model_changes[idx + 1] = {'action': 'replace',
                                       'layer': tf.keras.layers.Dense(weights.shape[-1],
                                                                      weights=[w2.numpy(), bias],
                                                                      activation=layer.get_config()['activation'],
                                                                      name='MovedDense',
                                                                      )}
        self.model_changes[idx] = {'action': 'insert',
                                   'layer': tf.keras.layers.Dense(units,
                                                                  weights=[w1.numpy()],
                                                                  activation=tf.keras.activations.linear,
                                                                  use_bias=False,
                                                                  name='InsertedDense')}

        self.update_model()
        if self.fine_tune:
            self.logger.info('Fine-tuning the optimized model.')
            self.model.fit(self.dataset, epochs=self.epochs)
        self.logger.info('Finished compression')


# class BinaryWeight(tf.keras.constraints.Constraint):
#     """
#     Contraint that rounds the weights to the closest value between 0 and 1.
#     """
#
#     def __init__(self, max_value):
#         super(BinaryWeight, self).__init__()
#
#     def __call__(self, weights):
#         return tf.round(tf.clip_by_value(weights, 0.0, 1.0))

class BinaryWeight(tf.keras.constraints.Constraint):
    """
    Contraint that rounds the weights to the closest value between 0 and 1.
    """

    def __init__(self, max_value):
        super(BinaryWeight, self).__init__()
        self.max_value = max_value

    def __call__(self, weights):
        return tf.round(tf.clip_by_norm(weights, self.max_value, axes=0))


class InsertDenseSparse(ModelCompression):
    """
    Compression technique that inserts a Dense layer inbetween two Dense layers.
    By inserting a smaller layer with C units, the number of weights is reduced
    from MxN to (M+N)xC. The weights are obtained by fitting a neural network to
    predict the same output as the original model.
    """

    def __init__(self, **kwargs):
        super(InsertDenseSparse, self).__init__(**kwargs)

    def compress_layer(self, layer_name, iterations=1000, verbose=False):
        self.logger.info('Searching for layer: {}'.format(layer_name))

        # Find layer
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        weights, bias = layer.get_weights()
        weights = tf.Variable(weights, dtype='float32')
        units = weights.shape[0] // 6
        w_init = tf.random_normal_initializer()
        basis = tf.Variable(name='basis',
                            initial_value=w_init(shape=(weights.shape[0], units)))

        sparse_dict = tf.Variable(name='sparse_code',
                                  initial_value=np.random.randint(0, 2, size=(units, weights.shape[-1])),
                                  constraint=BinaryWeight(units),
                                  dtype='float32')

        optimizer = tf.keras.optimizers.Adam()
        for i in range(iterations):
            with tf.GradientTape() as tape:
                pred = tf.matmul(basis, sparse_dict)
                loss = tf.reduce_sum(tf.nn.l2_loss(weights - pred))

            gradients = tape.gradient(loss, [basis, sparse_dict])
            optimizer.apply_gradients(zip(gradients, [basis, sparse_dict]))
            if verbose and i % 10000 == 0:
                print('Epoch {} Loss: {}'.format(i, loss))

        self.model_changes[idx + 1] = {'action': 'replace',
                                       'layer': tf.keras.layers.Dense(weights.shape[-1],
                                       weights=[sparse_dict.numpy(), bias],
                                       activation=layer.get_config()['activation'],
                                       name='SparseCodeLayer',
                                       kernel_constraint=BinaryWeight(units)
                                       )}
        self.model_changes[idx] = {'action': 'insert',
                                   'layer': tf.keras.layers.Dense(units,
                                       weights=[basis.numpy()],
                                       activation=tf.keras.activations.linear,
                                       use_bias=False,
                                       name='BasisDictLayer')}

        self.update_model()
        if self.fine_tune:
            self.logger.info('Fine-tuning the optimized model.')
            self.model.fit(self.dataset, epochs=self.epochs)
        self.logger.info('Finished compression')


    def compress_layers(self, layers, activation='relu', iterations=100, verbose=0):

        for x, y in self.dataset:
            for idx, layer in enumerate(self.model.layers):
                self.logger.info('Current layer: {}'.format(layer.name))
                if layer.name in layers:
                    self.logger.info('Compressing layer.')
                    output = layer(x)
                    weights, bias = layer.get_weights()
                    weights = tf.Variable(weights, dtype='float32')
                    units = weights.shape[0] // 6
                    w_init = tf.random_normal_initializer()
                    basis = tf.Variable(name='basis',
                                        initial_value=w_init(shape=(weights.shape[0], units)))

                    sparse_dict = tf.Variable(name='sparse_code',
                                              initial_value=np.random.randint(0, 2, size=(units, weights.shape[-1])),
                                              constraint=BinaryWeight(units),
                                              dtype='float32')

                    optimizer = tf.keras.optimizers.Adam()
                    for i in range(iterations):
                        with tf.GradientTape() as tape:
                            pred = tf.matmul(basis, sparse_dict)
                            loss = tf.reduce_sum(tf.nn.l2_loss(weights - pred))

                        gradients = tape.gradient(loss, [basis, sparse_dict])
                        optimizer.apply_gradients(zip(gradients, [basis, sparse_dict]))
                        if i % 1000 == 0:
                            print('Epoch {} Loss: {}'.format(i, loss))
                            print('{} ones of {}'.format(tf.reduce_sum(sparse_dict),
                                                         sparse_dict.shape[0] * sparse_dict.shape[1]))
                    break

                x = layer(x)
            break

        layers = copy.deepcopy(self.model.layers)
        dense0 = tf.keras.layers.Dense(units,
                                       weights=[basis.numpy()],
                                       activation=tf.keras.activations.linear,
                                       use_bias=False,
                                       name='BasisDictLayer')

        dense1 = tf.keras.layers.Dense(weights.shape[-1],
                                       weights=[sparse_dict.numpy(), bias],
                                       activation=activation,
                                       name='SparseCodeLayer',
                                       kernel_constraint=BinaryWeight(units)
                                       )
        layers[idx] = dense0
        layers.insert(idx + 1, dense1)

        self.logger.info('Finished compression')
        return layers


class InsertSVDConv(ModelCompression):
    """
    Compression techniques that reduces the number of weights in a filter by
    applying the filter in two steps, one horizontal and one vertical. Instead of
    using a DxD filter, a Dx1 is used followed by a 1xD filter. Thus, reducing the
    number of required weights. The process to find the weights of the one
    dimensional filters is by regressing a convolutional neural network and
    setting the output of the original filter as the target of the regression
    model.
    """

    def __init__(self, **kwargs):
        super(InsertSVDConv, self).__init__(**kwargs)

    def compress_layer(self, layer_name, iterations=5):
        self.logger.info('Searching for layer: {}'.format(layer_name))

        # Find layer
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        filters = layer.get_config()['filters']
        kernel_size = layer.get_config()['kernel_size']
        svd_model = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(filters // 6, kernel_size, activation='relu', name='SVDConv1',
                                    padding='same'),
             tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', name='SVDConv2')]
        )

        x = self.model.input
        output = self.model.layers[idx-1].output
        generate_input = tf.keras.Model(x, output)

        output = self.model.layers[idx].output
        generate_output = tf.keras.Model(x, output)

        svd_model.compile(optimizer='adam', loss='mae')
        for i in range(1,iterations+1):
            for x,y in self.dataset:
                x_in = generate_input(x)
                y = generate_output(x)
                loss = svd_model.train_on_batch(x_in,y)
            self.logger.info('Iteration={} has loss={}.'.format(i,loss))

        self.model_changes[idx + 1] = {'action': 'replace',
                                       'layer': svd_model.layers[1]}
        self.model_changes[idx] = {'action': 'insert',
                                   'layer': svd_model.layers[0]}

        self.update_model()
        if self.fine_tune:
            self.logger.info('Fine-tuning the optimized model.')
            self.model.fit(self.dataset, epochs=self.epochs)
        self.logger.info('Finished compression')


class DepthwiseSeparableConvolution(ModelCompression):
    """
    Compression techniques that reduces the number of weights in a filter by
    applying the filter in two steps, one horizontal and one vertical. Instead of
    using a DxD filter, a Dx1 is used followed by a 1xD filter. Thus, reducing the
    number of required weights. The process to find the weights of the one
    dimensional filters is by regressing a convolutional neural network and
    setting the output of the original filter as the target of the regression
    model.
    """

    def __init__(self, **kwargs):
        super(DepthwiseSeparableConvolution, self).__init__(**kwargs)

    def compress_layer(self, layer_name, iterations=5):
        self.logger.info('Searching for layer: {}'.format(layer_name))

        # Find layer
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        filters = layer.get_config()['filters']
        kernel_size = layer.get_config()['kernel_size']
        svd_model = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(filters=filters,
                                                                         kernel_size=(3, 3),
                                                                         name='DepthwiseSeparableLayer')])

        x = self.model.input
        output = self.model.layers[idx-1].output
        generate_input = tf.keras.Model(x, output)

        output = self.model.layers[idx].output
        generate_output = tf.keras.Model(x, output)

        svd_model.compile(optimizer='adam', loss='mae')
        for i in range(1,iterations+1):
            for x,y in self.dataset:
                x_in = generate_input(x)
                y = generate_output(x)
                loss = svd_model.train_on_batch(x_in,y)
            self.logger.info('Iteration={} has loss={}.'.format(i,loss))

        self.model_changes[idx] = {'action': 'replace',
                                   'layer': svd_model.layers[0]}

        self.update_model()
        if self.fine_tune:
            self.logger.info('Fine-tuning the optimized model.')
            self.model.fit(self.dataset, epochs=self.epochs)
        self.logger.info('Finished compression')

class FireLayer(tf.keras.Model):
    def __init__(self, s1x1, e1x1, e3x3, name, **kwargs):
        super(FireLayer, self).__init__(**kwargs)

        self.squeeze = tf.keras.layers.Conv2D(s1x1, (1, 1), activation='relu', name=name + '/squeeze')
        self.expand1x1 = tf.keras.layers.Conv2D(e1x1, (1, 1), padding='valid', name=name + '/expand1x1')
        self.expand3x3 = tf.keras.layers.Conv2D(e3x3, (3, 3), padding='same', name=name + '/expand3x3')
        w_init = tf.initializers.constant(1)

    def call(self, inputs):
        x = self.squeeze(inputs)
        o1x1 = self.expand1x1(x)
        o3x3 = self.expand3x3(x)
        x = tf.keras.layers.concatenate([o1x1, o3x3], axis=3)
        x = tf.keras.layers.Cropping2D(cropping=1)(x)
        return x


class FireLayerCompression(ModelCompression):
    """
    Compression techniques that replaces a convolutional layer by a fire layer,
    which consists of 1x1 and 3x3 convolutions. A 1x1 is
    """

    def __init__(self, **kwargs):
        super(FireLayerCompression, self).__init__(**kwargs)

    def compress_layer(self, layer_name, iterations=5):
        self.logger.info('Searching for layer: {}'.format(layer_name))

        # Find layer
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        filters = layer.get_config()['filters']
        svd_model = tf.keras.Sequential(
            [FireLayer(s1x1=filters, e1x1=filters // 2, e3x3=filters // 2, name='FireLayer')])

        x = self.model.input
        output = self.model.layers[idx-1].output
        generate_input = tf.keras.Model(x, output)

        output = self.model.layers[idx].output
        generate_output = tf.keras.Model(x, output)

        svd_model.compile(optimizer='adam', loss='mae')
        for i in range(1,iterations+1):
            for x,y in self.dataset:
                x_in = generate_input(x)
                y = generate_output(x)
                loss = svd_model.train_on_batch(x_in,y)
            self.logger.info('Iteration={} has loss={}.'.format(i,loss))

        self.model_changes[idx] = {'action': 'replace',
                                   'layer': svd_model.layers[0]}

        self.update_model()
        if self.fine_tune:
            self.logger.info('Fine-tuning the optimized model.')
            self.model.fit(self.dataset, epochs=self.epochs)
        self.logger.info('Finished compression')


class MLPConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation=None, *args, **kwargs):
        super(MLPConv, self).__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        batch, h, w, channels = input_shape
        w_init = tf.random_normal_initializer()
        self.w_0 = tf.Variable(name="kernel0",
                               initial_value=w_init(shape=(tf.reduce_prod(self.kernel_size) * channels, self.filters),
                                                    dtype='float32'),
                               trainable=True)
        self.w_1 = tf.Variable(name="kernel1",
                               initial_value=w_init(shape=(self.filters, self.filters),
                                                    dtype='float32'),
                               trainable=True)
        b_init = tf.zeros_initializer()
        self.b_0 = tf.Variable(name="bias0",
                               initial_value=b_init(shape=(self.filters,), dtype='float32'),
                               trainable=True)
        self.b_1 = tf.Variable(name="bias1",
                               initial_value=b_init(shape=(self.filters,), dtype='float32'),
                               trainable=True)
        fh = self.kernel_size[0]
        fw = self.kernel_size[1]
        self.indexes = tf.constant([[x, y] for x in range(fh // 2, h - 1) for y in range(fw // 2, w - 1)])
        super().build(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, inputs):
        batch, h, w, channels = inputs.shape

        fh, fw = self.kernel_size
        dh = fh // 2
        dw = fw // 2

        def calculate_output(x, y, inputs, kernel_size, w0, b0, w1, b1, activation):
            activation = tf.keras.activations.get(activation)
            batch, h, w, channels = inputs.shape
            fh = kernel_size[0]
            fw = kernel_size[1]
            dh = fh // 2
            dw = fw // 2
            reshaped_input = tf.reshape(inputs[:, x - dh:x + dh + 1, y - dw:y + dw + 1, :], shape=[batch, -1])
            next_input = activation(tf.matmul(reshaped_input, w0) + b0)
            return activation(tf.matmul(next_input, w1) + b1)

        func = lambda position: calculate_output(position[0], position[1],
                                                 inputs,
                                                 self.kernel_size,
                                                 self.w_0, self.b_0,
                                                 self.w_1, self.b_1,
                                                 self.activation)

        output = tf.vectorized_map(func, elems=self.indexes)
        output = tf.transpose(output, perm=[2, 0, 1])
        output = tf.reshape(output, shape=(batch, h - 2, w - 2, self.filters))
        return output


class MLPCompression(ModelCompression):
    """
    Compression techniques that replaces a convolutional layer by a Multi-layer
    perceptron that learns to generate the output of each filter.
    """

    def __init__(self, **kwargs):
        super(MLPCompression, self).__init__(**kwargs)


    def compress_layer(self, layer_name, iterations=5):
        self.logger.info('Searching for layer: {}'.format(layer_name))

        # Find layer
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        filters = layer.get_config()['filters']
        kernel_size = layer.get_config()['kernel_size']
        activation = layer.get_config()['activation']
        mlp_model = tf.keras.Sequential([MLPConv(kernel_size=kernel_size,
                                                 filters=filters,
                                                 activation=activation,
                                                 name='MLPConv')])

        x = self.model.input
        output = self.model.layers[idx-1].output
        generate_input = tf.keras.Model(x, output)

        output = self.model.layers[idx].output
        generate_output = tf.keras.Model(x, output)

        mlp_model.compile(optimizer='adam', loss='mae')
        for i in range(1,iterations+1):
            for x,y in self.dataset:
                x_in = generate_input(x)
                y = generate_output(x)
                loss = mlp_model.train_on_batch(x_in,y)
            self.logger.info('Iteration={} has loss={}.'.format(i,loss))

        self.model_changes[idx] = {'action': 'replace',
                                   'layer': mlp_model.layers[0]}

        self.update_model()
        if self.fine_tune:
            self.logger.info('Fine-tuning the optimized model.')
            self.model.fit(self.dataset, epochs=self.epochs)
        self.logger.info('Finished compression')

class SparseConnectionConv2D(tf.keras.layers.Conv2D):
    def __init__(self, sparse_connections, *args, **kwargs):
        super(SparseConnectionConv2D, self).__init__(*args, **kwargs)
        self.sparse_connections = sparse_connections

    def convolution_op(self, inputs, kernel):
        """
        Exact copy from the method of Conv class.
        """
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding

        return tf.nn.convolution(
            inputs,
            kernel,
            strides=list(self.strides),
            padding=tf_padding,
            dilations=list(self.dilation_rate),
            data_format=self._tf_data_format,
            name=self.__class__.__name__)

    def call(self, inputs):
        batch, h, w, channels = inputs.shape
        temp_weights = self.kernel.numpy()
        for idx in range(self.filters):
            filter_connections = np.where(
                np.asarray(self.sparse_connections[idx * channels:idx * channels + channels]) == 0)
            temp_weights[:, :, filter_connections, idx] = 0.0

        new_weights = tf.Variable(temp_weights, dtype='float32')

        return self.convolution_op(inputs, new_weights)


def evaluation_function(ind, model, loss_object, dataset, layers):
    loss = []
    for x, y in dataset:
        for layer in model.layers:
            if layer.name in layers:
                temp = layer(x)
                config = layer.get_config()
                sparse_c_layer = SparseConnectionConv2D(sparse_connections=ind, filters=config['filters'],
                                                        kernel_size=config['kernel_size'])
                sparse_output = sparse_c_layer(x)
                sparse_c_layer.set_weights(layer.get_weights())
                x = sparse_c_layer(x)
            else:
                x = layer(x)
        loss.append(loss_object(y, x))
    return tf.reduce_sum(loss).numpy(),


class GeneticAlgorithm:
    def __init__(self, eval_func, model, dataset, layers, loss_object, mut_bit=0.1, population=6, generations=1,
                 cxpb=0.5,
                 mutpb=0.5):
        self.model = model
        self.layers = layers
        self.dataset = dataset
        self.population = population
        self.ngen = generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.toolbox = base.Toolbox()
        # self.toolbox.register('select', tools.selNSGA2)
        # creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
        self.toolbox.register('select', tools.selTournament, tournsize=2)
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        self.toolbox.register('mate', tools.cxTwoPoint)
        self.toolbox.register('mutate', tools.mutFlipBit, indpb=mut_bit)

        self.toolbox.register('evaluate', evaluation_function, model=self.model, dataset=self.dataset,
                              layers=self.layers, loss_object=loss_object)

        creator.create('Individual', list, fitness=creator.FitnessMin)
        self.toolbox.register('bit', random.randint, a=0, b=1)
        self.toolbox.register('individual', tools.initRepeat, creator.Individual, self.toolbox.bit,
                              n=self.calculate_num_filters())
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register('min', np.min)
        self.stats.register('max', np.max)
        self.stats.register('mean', np.mean)
        self.stats.register('std', np.std)
        self.stats.register('median', np.median)

        self.hof = tools.HallOfFame(10)

    def optimize(self):
        pop = self.toolbox.population(n=self.population)
        population, log = algorithms.eaMuPlusLambda(pop, self.toolbox,
                                                    cxpb=self.cxpb,
                                                    mu=self.population,
                                                    lambda_=self.population // 2,
                                                    mutpb=self.mutpb,
                                                    ngen=self.ngen,
                                                    stats=self.stats,
                                                    halloffame=self.hof,
                                                    verbose=True)

        return self.hof

    def calculate_num_filters(self):
        for x, y in self.dataset:
            for idx, layer in enumerate(self.model.layers):
                if layer.name in self.layers:
                    filters = layer.get_config()['filters']
                    return filters * x.shape[-1]
                x = layer(x)
            break


class SparseConnectionsCompression(ModelCompression):
    """
    Compression technique that sparsifies the connections between input channels
    and output channels when applying filters in a convolutional layer.
    """

    def __init__(self, model, dataset, loss_object):
        super(SparseConnectionsCompression, self).__init__(model, dataset)
        self.loss_object = loss_object

    def compress_layers(self, layers, population=10, generations=5):
        self.logger.info('Setting up genetic algorithm.')
        ga = GeneticAlgorithm(eval_func=evaluation_function,
                              population=population,
                              generations=generations,
                              model=self.model,
                              dataset=self.dataset,
                              layers=layers,
                              loss_object=self.loss_object)
        self.logger.info('Running genetic algorithm.')
        results = ga.optimize()

        evaluations = list(map(lambda x: x.fitness.values, results))
        self.logger.info('Top 10: {}'.format(evaluations))
        best = np.argmin(evaluations)
        self.logger.info('Searching layer to replace.')
        for x, y in self.dataset:
            for idx, layer in enumerate(self.model.layers):
                config = layer.get_config()
                if layer.name in layers:
                    self.logger.info('Replacing layer {} with {}'.format(config['name'], self.get_technique()))
                    filters = config['filters']
                    kernel = config['kernel_size']
                    weights, bias = layer.get_weights()
                    sparse_conv_layer = SparseConnectionConv2D(sparse_connections=results[best], filters=filters,
                                                               kernel_size=kernel, name='SparseConnectionsConv')
                    _ = sparse_conv_layer(x)
                    sparse_conv_layer.set_weights([weights, bias])
                    break
                x = layer(x)
            break

        layers = copy.deepcopy(self.model.layers)
        layers[idx] = sparse_conv_layer
        self.logger.info('Layer {} is {}'.format(idx, layers[idx].name))
        self.logger.info('Chromosome: {}'.format(results[best]))
        self.logger.info('Evaluation: {}'.format(evaluations[best]))
        self.logger.info('Number of connections: {}'.format(sum(results[best])))
        self.logger.info('Finished compression')
        return layers


class SparseConvolution2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters, number_bases, activation=None, *args, **kwargs):
        super(SparseConvolution2D, self).__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.activation = tf.keras.activations.get(activation)
        self.bases = number_bases

    def build(self, input_shape):
        batch, h, w, channels = input_shape
        identity_initializer = tf.initializers.Identity()
        w_init = tf.random_normal_initializer()
        self.s = tf.Variable(name="S",
                             initial_value=w_init(shape=(self.bases, self.filters),
                                                  dtype='float32'),
                             trainable=True)
        kh, kw = self.kernel_size.shape
        self.q = tf.Variable(name="Q",
                             initial_value=identity_initializer(shape=(kh, kw, self.filters),
                                                                dtype='float32'),
                             trainable=True)

        self.p = tf.Variable(name="P",
                             initial_value=identity_initializer(shape=(channels, channels),
                                                                dtype='float32'),
                             trainable=True)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
                             initial_value=b_init(shape=(self.filters,), dtype='float32'),
                             trainable=True)

        fh = self.kernel_size[0]
        fw = self.kernel_size[1]
        self.indexes = tf.constant([[x, y] for x in range(fh // 2, h - 1) for y in range(fw // 2, w - 1)])
        super().build(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, inputs):
        batch, h, w, channels = inputs.shape
        fh, fw = self.kernel_size

        def calculate_output(x, y, inputs, kernel_size, w0, b0, w1, b1, activation):
            activation = tf.keras.activations.get(activation)
            batch, h, w, channels = inputs.shape
            fh = kernel_size[0]
            fw = kernel_size[1]
            dh = fh // 2
            dw = fw // 2
            reshaped_input = tf.reshape(inputs[:, x - dh:x + dh + 1, y - dw:y + dw + 1, :], shape=[batch, -1])
            next_input = activation(tf.matmul(reshaped_input, w0) + b0)
            return activation(tf.matmul(next_input, w1) + b1)

        func = lambda position: calculate_output(position[0], position[1],
                                                 inputs,
                                                 self.kernel_size,
                                                 self.w_0, self.b_0,
                                                 self.w_1, self.b_1,
                                                 self.activation)

        output = tf.vectorized_map(func, elems=self.indexes)
        output = tf.transpose(output, perm=[2, 0, 1])
        output = tf.reshape(output, shape=(batch, h - 2, w - 2, self.filters))
        return output


class SparseConvolutionCompression(ModelCompression):
    """
    Compression technique that performs an sparse convolution
    """

    def __init__(self, model, dataset):
        super(SparseConvolutionCompression, self).__init__(model, dataset)

    def compress_layers(self, layers):
        self.logger.info('Searching layer to replace.')
        for x, y in self.dataset:
            for idx, layer in enumerate(self.model.layers):
                config = layer.get_config()
                if layer.name in layers:
                    self.logger.info('Replacing layer {} with {}'.format(config['name'], self.get_technique()))
                    filters = config['filters']
                    kernel = config['kernel_size']
                    weights, bias = layer.get_weights()
                    sparse_conv_layer = SparseConnectionConv2D(sparse_connections=results[best], filters=filters,
                                                               kernel_size=kernel, name='SparseConnectionsConv')
                    _ = sparse_conv_layer(x)
                    sparse_conv_layer.set_weights([weights, bias])
                    break
                x = layer(x)
            break

        layers = copy.deepcopy(self.model.layers)
        layers[idx] = sparse_conv_layer
        self.logger.info('Layer {} is {}'.format(idx, layers[idx].name))
        self.logger.info('Chromosome: {}'.format(results[best]))
        self.logger.info('Evaluation: {}'.format(evaluations[best]))
        self.logger.info('Number of connections: {}'.format(sum(results[best])))
        self.logger.info('Finished compression')
        return layers
