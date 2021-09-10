import tensorflow as tf
import numpy as np
import logging
import copy
import random
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

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.history = None
        self.logger = logging.getLogger(__name__)

    def get_technique(self):
        return self.__class__.__name__

    def get_model(self):
        return self.model

    def compress_layer(self, layer):
        pass


class DeepCompression(ModelCompression):
    """
    Compression technique that sets to 0 all weights that are below a threshold in
    a Dense layer. No clustering and Huffman encoding is performed.
    """

    def __init__(self, model, dataset):
        super(DeepCompression, self).__init__(model, dataset)

    def compress_layers(self, layers, threshold=0.0001):
        for x, y in self.dataset:
            for layer in self.model.layers:
                self.logger.info('Current layer: {}'.format(layer.name))
                if layer.name in layers:
                    self.logger.info('Compressing layer.')
                    output = layer(x)
                    weights = layer.get_weights()
                    tf_weights = tf.Variable(weights[0], trainable=True)
                    below_threshold = tf.abs(tf_weights) < threshold
                    self.logger.info('Pruned weights: {}'.format(tf.math.count_nonzero(below_threshold)))
                    new_weights = tf.where(below_threshold, 0, tf_weights)
                    layer.set_weights([new_weights, weights[1]])
                x = layer(x)
            break
        self.logger.info('Finished compression')
        return self.model.layers


class ReplaceDenseWithGlobalAvgPool(ModelCompression):
    """
    Compression technique that replaces a Dense layer with a Global Average
    Pooling layer.
    """

    def __init__(self, model, dataset):
        super(ReplaceDenseWithGlobalAvgPool, self).__init__(model, dataset)

    def compress_layers(self, layers):

        for idx, layer in enumerate(self.model.layers):
            self.logger.info('Current layer: {}'.format(layer.name))
            if layer.name in layers:
                self.logger.info('Replacing layer.')
                break

        layers = copy.deepcopy(self.model.layers)
        layers[idx] = tf.keras.layers.GlobalAveragePooling2D()
        # Remove previous Dense
        layers.pop(idx - 1)
        # Remove Flatten
        layers.pop(idx - 2)
        self.logger.info('Finished compression')
        return layers


class InsertDenseSVD(ModelCompression):
    """
    Compression technique that inserts a Dense layer inbetween two Dense layers.
    By inserting a smaller layer with C units, the number of weights is reduced
    from MxN to (M+N)xC. The weights are obtained by fitting a neural network to
    predict the same output as the original model.
    """

    def __init__(self, model, dataset):
        super(InsertDenseSVD, self).__init__(model, dataset)

    def compress_layers(self, layers, activation='relu', iterations=100, verbose=0):

        for x, y in self.dataset:
            for idx, layer in enumerate(self.model.layers):
                self.logger.info('Current layer: {}'.format(layer.name))
                if layer.name in layers:
                    self.logger.info('Compressing layer.')
                    output = layer(x)
                    weights, bias = layer.get_weights()
                    units = weights.shape[0] // 6
                    weights = tf.Variable(weights, dtype='float32')
                    w_init = tf.random_normal_initializer()
                    w1 = tf.Variable(name='w1',
                                     initial_value=w_init(shape=(weights.shape[0], units)))

                    w2 = tf.Variable(name='w1',
                                     initial_value=w_init(shape=(units, weights.shape[1])))

                    optimizer = tf.keras.optimizers.Adam()
                    for i in range(iterations):
                        with tf.GradientTape() as tape:
                            pred = tf.matmul(w1, w2)
                            loss = tf.reduce_sum(tf.nn.l2_loss(weights - pred))

                        gradients = tape.gradient(loss, [w1, w2])
                        optimizer.apply_gradients(zip(gradients, [w1, w2]))
                        if i % 10000 == 0:
                            print('Epoch {} Loss: {}'.format(i, loss))
                    break

                x = layer(x)
            break

        layers = copy.deepcopy(self.model.layers)
        dense0 = tf.keras.layers.Dense(units,
                                       weights=[w1.numpy()],
                                       activation=tf.keras.activations.linear,
                                       use_bias=False,
                                       name='InsertedDense')

        dense1 = tf.keras.layers.Dense(weights.shape[-1],
                                       weights=[w2.numpy(), bias],
                                       activation=activation,
                                       name='MovedDense',
                                       )
        layers[idx] = dense0
        layers.insert(idx + 1, dense1)

        self.logger.info('Finished compression')
        return layers


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

    def __init__(self, model, dataset):
        super(InsertDenseSparse, self).__init__(model, dataset)

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

    def __init__(self, model, dataset):
        super(InsertSVDConv, self).__init__(model, dataset)

    def compress_layers(self, layers, activation='relu', iterations=100, verbose=0):
        for x, y in self.dataset:
            for idx, layer in enumerate(self.model.layers):
                self.logger.info('Current layer: {}'.format(layer.name))
                if layer.name in layers:
                    self.logger.info('Inserting Convolution with 1/12 of filters.')
                    output = layer(x)

                    self.logger.info('Input shape: {}'.format(x.shape))
                    self.logger.info('Output shape: {}'.format(output.shape))
                    filters = layer.get_config()['filters']
                    kernel_size = layer.get_config()['kernel_size']
                    svd_model = tf.keras.Sequential(
                        [tf.keras.layers.Conv2D(filters // 6, kernel_size, activation='relu', name='SVDConv1',
                                                padding='same'),
                         tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', name='SVDConv2')]
                    )

                    svd_model.compile(optimizer='sgd', loss='mae')

                    svd_model.fit(x, output, batch_size=32, epochs=iterations, verbose=verbose)

                    break

                x = layer(x)
            break
        layers = copy.deepcopy(self.model.layers)
        weights = svd_model.layers[0].get_weights()
        conv0 = tf.keras.layers.Conv2D(filters // 6, kernel_size, weights=weights, activation='relu', name='SVDConv1',
                                       padding='same')
        weights = svd_model.layers[1].get_weights()
        conv1 = tf.keras.layers.Conv2D(filters, kernel_size, weights=weights, activation='relu', name='SVDConv2')
        layers[idx] = conv0
        layers.insert(idx + 1, conv1)
        self.logger.info('Finished compression')
        return layers


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

    def __init__(self, model, dataset):
        super(DepthwiseSeparableConvolution, self).__init__(model, dataset)

    def compress_layers(self, layers, activation='relu', iterations=100, verbose=0):
        for x, y in self.dataset:
            for idx, layer in enumerate(self.model.layers):
                self.logger.info('Current layer: {}'.format(layer.name))
                if layer.name in layers:
                    self.logger.info('Replacing layer by Depthwise convolution.')
                    output = layer(x)

                    self.logger.info('Input shape: {}'.format(x.shape))
                    self.logger.info('Output shape: {}'.format(output.shape))
                    filters = layer.get_config()['filters']
                    svd_model = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(filters=filters,
                                                                                     kernel_size=(3, 3),
                                                                                     name='DepthwiseSeparableLayer')])

                    svd_model.compile(optimizer='sgd', loss='mae')

                    svd_model.fit(x, output, batch_size=32, epochs=iterations, verbose=verbose)
                    break
                x = layer(x)
            break
        layers = copy.deepcopy(self.model.layers)
        weights = svd_model.layers[0].get_weights()
        conv0 = tf.keras.layers.SeparableConv2D(filters=filters,
                                                weights=weights,
                                                kernel_size=(3, 3),
                                                name='DepthwiseSeparableLayer')
        layers[idx] = conv0
        self.logger.info('Finished compression')
        return layers


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
        batch, h, w, channels = x.shape
        x = tf.keras.layers.Cropping2D(cropping=1)(x)
        return x


class FireLayerCompression(ModelCompression):
    """
    Compression techniques that replaces a convolutional layer by a fire layer,
    which consists of 1x1 and 3x3 convolutions. A 1x1 is
    """

    def __init__(self, model, dataset):
        super(FireLayerCompression, self).__init__(model, dataset)

    def compress_layers(self, layers, iterations=100, verbose=0):
        for x, y in self.dataset:
            for idx, layer in enumerate(self.model.layers):
                self.logger.info('Current layer: {}'.format(layer.name))
                if layer.name in layers:
                    self.logger.info('Replacing layer by Fire Layer.')
                    output = layer(x)

                    self.logger.info('Input shape: {}'.format(x.shape))
                    self.logger.info('Output shape: {}'.format(output.shape))
                    filters = layer.get_config()['filters']
                    svd_model = tf.keras.Sequential(
                        [FireLayer(s1x1=filters, e1x1=filters // 2, e3x3=filters // 2, name='FireLayer')])

                    svd_model.compile(optimizer='sgd', loss='mae')

                    svd_model.fit(x, output, batch_size=32, epochs=iterations, verbose=verbose)

                    break
                x = layer(x)
            break
        new_layers = svd_model.layers
        layers = copy.deepcopy(self.model.layers)
        layers[idx] = new_layers[0]
        self.logger.info('Finished compression')
        return layers


class MLPConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size,  activation=None, *args, **kwargs):
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

    def __init__(self, model, dataset):
        super(MLPCompression, self).__init__(model, dataset)

    def compress_layers(self, layers, iterations=100, verbose=0):
        for x, y in self.dataset:
            for idx, layer in enumerate(self.model.layers):
                self.logger.info('Current layer: {}'.format(layer.name))
                if layer.name in layers:
                    self.logger.info('Replacing layer with a MLPConv net.')
                    output = layer(x)

                    self.logger.info('Input shape: {}'.format(x.shape))
                    self.logger.info('Output shape: {}'.format(output.shape))
                    filters = layer.get_config()['filters']
                    kernel_size = layer.get_config()['kernel_size']
                    activation = layer.get_config()['activation']
                    mlp_model = tf.keras.Sequential([MLPConv(kernel_size=kernel_size,
                                                             filters=filters,
                                                             activation=activation,
                                                             name='MLPConv')])

                    mlp_model.compile(optimizer='adam', loss='mse')

                    mlp_model.fit(x, output, batch_size=32, epochs=iterations, verbose=verbose)
                    break
                x = layer(x)
            break
        new_layers = mlp_model.layers
        layers = copy.deepcopy(self.model.layers)
        layers[idx] = new_layers[0]
        self.logger.info('Finished compression')
        return layers


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
        outputs = self.convolution_op(inputs, self.kernel).numpy()
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