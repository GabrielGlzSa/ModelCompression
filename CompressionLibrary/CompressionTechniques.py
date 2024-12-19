# uncompyle6 version 3.8.0callback
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59) 
# [GCC 7.5.0]
# Embedded file name: ./ModelCompression/CompressionTechniques.py
# Compiled at: 2022-05-16 17:17:52
# Size of source mod 2**32: 57841 bytes

import math, tensorflow as tf, tensorflow.keras.backend as K, numpy as np, logging
from datetime import datetime
from CompressionLibrary.custom_callbacks import AddSparseConnectionsCallback
from CompressionLibrary.custom_constraints import kNonZeroes, SparseWeights
from CompressionLibrary.custom_layers import DenseSVD, SparseSVD, FireLayer, MLPConv, ConvSVD, SparseConvolution2D, SparseConnectionsConv2D
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cluster import KMeans

class ModelCompression:
    __doc__ = '\n    Base class for compressing a deep learning model. The class takes a tensorflow\n    model and a dataset that will be used to fit a regression.\n    '

    def __init__(self, model, optimizer, loss, metrics, input_shape, dataset, fine_tuning=False, tuning_verbose=1, tuning_epochs=10, num_batches=None, callbacks = None, strategy=None):
        """

        :param model: tensorflow model that will be optimized.
        :param optimizer: optimizer that will be used to compile the optimized model.
        :param loss: loss object that will be used to compile the optimized model.
        :param metrics: metrics that will be used to compile the optimized model.
        :param dataset: dataset that will be used for compressing a layer and/or fine-tuning.
        :param fine_tuning: flag to select if the optimized model will be trained.
        :param tuning_epochs: number of epochs of the fine-tuning.
        :param num_batches: number of batches to be used for compression.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss
        self.metrics = metrics
        self.dataset = dataset
        self.fine_tuning = fine_tuning
        self.input_shape = input_shape
        self.tuning_epochs = tuning_epochs
        self.logger = logging.getLogger(__name__)
        self.model_changes = {}
        self.target_layer_type = None
        self.tuning_verbose = tuning_verbose
        self.new_layer_name = None
        self.num_batches = num_batches
        self.callbacks = callbacks
        self.strategy = strategy
        if num_batches is not None:
            self.dataset = self.dataset.take(num_batches)

    def get_technique(self):
        """
        Returns the name of the compression technique.
        :return: name of the class.
        """
        return self.__class__.__name__

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

    def get_new_layer(self, old_layer):
        pass

    def replace_layer(self, new_layer, layer_name):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:]:
            if layer.name == layer_name:
                x = new_layer(x)
            else:
                x = layer(x)
            
        return tf.keras.Model(inputs, x)

    def compress_layer(self, layer_name: str, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.logger.debug(f'Using method {self.get_technique()} to compress {layer_name}.')
        idx = self.find_layer(layer_name)
        layer = self.model.layers[idx]
        # Create the new layer that will replace the target layer.


        new_layer, new_layer_name, layer_weights_before, layer_weight_after = self.get_new_layer(layer)
     

        # Create the new model.
        self.model = self.replace_layer(new_layer, layer_name)

        self.model.compile(optimizer=self.optimizer, loss=self.loss_object, metrics=self.metrics)

        self.new_layer_name = new_layer_name
        
        fake_input = tf.zeros(shape=self.input_shape, dtype=tf.float32)
        fake_input = tf.expand_dims(fake_input, axis=0)
        self.model(fake_input)

        
        if self.fine_tuning:
            self.model.fit(self.dataset, epochs=self.tuning_epochs, callbacks=self.callbacks, verbose=self.tuning_verbose)

            if self.__class__.__name__ == 'SparseConnectionsCompression':
                idx = self.find_layer(new_layer_name)
                layer = self.model.layers[idx]
                kernel_size = layer.get_config()['kernel_size']
                connections = layer.get_connections()
                num_zeroes = len(connections) - np.sum(connections)
                if isinstance(kernel_size, int):
                    channel_weights = kernel_size**2
                else:
                    channel_weights = tf.reduce_prod(kernel_size)


                layer_weight_after = layer_weights_before - (channel_weights * num_zeroes)

        # Get how many weights were removed.
        w_diff = layer_weights_before - layer_weight_after

        self.logger.debug(f'Removed layer {layer_name} had {layer_weights_before} weights. New layer has {layer_weight_after} weights. Difference is {w_diff}')


        self.logger.debug('Finished compression')

class DeepCompression(ModelCompression):
    __doc__ = '\n    Compression technique that sets to 0 all weights that are below a threshold in\n    a Dense layer. No clustering and Huffman encoding is performed.\n    '

    def __init__(self, **kwargs):
        (super(DeepCompression, self).__init__)(**kwargs)
        self.target_layer_type = 'dense'

    def get_new_layer(self, old_layer):
        weights, bias = old_layer.get_weights()
        config = old_layer.get_config()
        units = config['units']
        activation = config['activation']
        # Get weights that are below threshold.
        below_threshold = tf.abs(weights) < self.threshold
        # Replace with 0 all weights below threshold.
        new_weights = tf.where(below_threshold, 0.0, weights)

        num_zeroes = tf.math.count_nonzero(tf.abs(new_weights[0]) == 0.0).numpy()
        weights_before = np.sum([K.count_params(w) for w in old_layer.trainable_weights])
        weights_after = weights_before - num_zeroes

        new_layer = tf.keras.layers.Dense(units=units,
                  activation=activation,
                  name=old_layer.name + '/DeepComp',
                  kernel_constraint=SparseWeights(self.threshold))

        new_layer(old_layer.input)

        new_layer.set_weights([new_weights, bias])
        

        self.logger.debug(f'Replaced layer was using {weights_before} weights.')
        self.logger.debug(f'DeepComp  is using {weights_after} weights.')

        return new_layer, new_layer.name, weights_before, weights_after

class ReplaceDenseWithGlobalAvgPool(ModelCompression):
    __doc__ = '\n    Compression technique that replaces all dense and flatten layers\n    between the last convolutional layer and the softmax layer with a GlobalAveragePooling2D layer.\n    '

    def __init__(self, **kwargs):
        (super(ReplaceDenseWithGlobalAvgPool, self).__init__)(**kwargs)
        self.target_layer_type = 'dense'

    def replace_layer(self, new_layer, layer_name):
        """
        New layer is a list that has Global Avg Pooling and Output layer. 
        """
        flatten_idx = self.find_layer('flatten')
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        if isinstance(self.model.layers[0], tf.keras.layers.InputLayer):
            x = self.model.layers[1](inputs)
            start = 2
        else:
            x = self.model.layers[0](inputs)
            start = 1
        for layer in self.model.layers[start:flatten_idx]:
            x = layer(x)

        for layer in new_layer:
            x = layer(x)
            
        return tf.keras.Model(inputs, x)


    def get_new_layer(self, old_layer):
        self.logger.debug('Searching for all dense layers.')
        flatten_idx = self.find_layer('flatten')
        conv_layer_config = self.model.layers[flatten_idx - 2].get_config()
        filters = conv_layer_config['filters']
        removed_weights = 0
        removed_layers = self.model.layers[flatten_idx+1:-1]
        # Calculate the number of weights that were removed for the hidden dense layers.
        for layer in removed_layers:
            layer_removed_weights = np.sum([K.count_params(w) for w in layer.trainable_weights])
            self.logger.debug(f'Removing {layer.name} with {layer_removed_weights} weights.')
            removed_weights += layer_removed_weights
         # Calculate the number of weights that were removed for the output dense layer.
        softmax_layer =  self.model.layers[-1]
        num_classes = softmax_layer.get_config()['units']
        softmax_removed_weights = np.sum([K.count_params(w) for w in softmax_layer.trainable_weights])
        self.logger.debug(f'Removing {softmax_layer.name} with {softmax_removed_weights} weights.')


        new_layer_name = old_layer.name + '/GlobalAvgPool'
        global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D(name=new_layer_name)
        new_softmax_layer = tf.keras.layers.Dense(num_classes, input_shape=[filters], activation='softmax', name=softmax_layer.name+'/GAP')

        weights_before = removed_weights

        # Number of weights is weights + bias
        weights_after =  num_classes * filters + num_classes

        return [global_avg_pooling, new_softmax_layer], new_layer_name, weights_before, weights_after


class InsertDenseSVD(ModelCompression):
    __doc__ = '\n    Compression technique that inserts a smaller dense layer using Singular Value Decomposition. By\n    inserting a smaller layer with C units, the number of weights is reduced\n    from MxN to (M+N)xC. The weights are obtained by fitting a neural network to\n    predict the same output as the original model.\n    '

    def __init__(self, **kwargs):
        (super(InsertDenseSVD, self).__init__)(**kwargs)
        self.target_layer_type = 'dense'

    def get_new_layer(self, old_layer):
        weights, bias = old_layer.get_weights()
        input_size , units = weights.shape
        if self.percentage is None:
            if self.hidden_units is None:
                hidden_units = units//12
                self.percentage = 100 * hidden_units / units
            else:
                hidden_units = self.hidden_units
                self.percentage = 100 * hidden_units / units
        else:
            max_units = (input_size * units)//(input_size+units)
            if isinstance(self.percentage, int) or np.issubdtype(self.percentage, np.integer):
                hidden_units = math.ceil(max_units * (self.percentage/100))
            elif isinstance(self.percentage, float) or isinstance(self.percentage, np.floating):
                # Max number of hidden units in order to have almost the same number of weights.
                hidden_units = math.ceil(max_units*self.percentage)
                self.percentage *= 100
        
        activation = old_layer.get_config()['activation']
        
        self.logger.debug(f'SVD is being calculated for shape {weights.shape} using {hidden_units} singular values ({self.percentage}%).')
        try:
            s, u, v = tf.linalg.svd(weights, full_matrices=False)
        except Exception as e:
            print(e)
            with tf.device('/CPU:0'):
                s, u, v = tf.linalg.svd(weights, full_matrices=False)
                
        u = tf.slice(u, begin=[0, 0], size=[weights.shape[0], hidden_units])
        s = tf.slice(s, begin=[0], size=[hidden_units])

        # Transpose V as V is returned as V^T.
        v = tf.slice(tf.linalg.matrix_transpose(v), begin=[0, 0], size=[hidden_units, units])
        n = tf.matmul(tf.linalg.diag(s), v)
        new_weights = tf.matmul(u, n)
        loss = tf.reduce_mean(tf.square(weights - new_weights))
        self.logger.debug(f'New weights have MSE of {loss}.')

        new_layer = DenseSVD(units=units, hidden_units=hidden_units,
                  activation=activation,
                  name=old_layer.name + '/DenseSVD')

        new_layer(old_layer.input)

        new_layer.set_weights([u.numpy(), n, bias])
        
        weights_before = np.sum([K.count_params(w) for w in old_layer.trainable_weights])
        weights_after = np.sum([K.count_params(w) for w in new_layer.trainable_weights])

        self.logger.debug(f'Replaced layer was using {weights_before} weights.')
        self.logger.debug(f'DenseSVD  is using {weights_after} weights.')

        return new_layer, new_layer.name, weights_before, weights_after


class InsertDenseSparse(ModelCompression):
    __doc__ = '\n    Compression technique that inserts a Dense layer inbetween two Dense layers.\n    By inserting a smaller layer with C units, the number of weights is reduced\n    from MxN to (M+N)xC. The weights are obtained by fitting a neural network to\n    predict the same output as the original model.\n    '

    def __init__(self, **kwargs):
        (super(InsertDenseSparse, self).__init__)(**kwargs)
        self.target_layer_type = 'dense'

    
    def get_new_layer(self, old_layer):
        weights, bias = old_layer.get_weights()
        features, units = weights.shape
        activation=old_layer.get_config()['activation']
        # Generalist paper uses 1/6 and 1/3 respectively.
        n_basis_vectors = units//4
        max_basis_vectors = n_basis_vectors//2

        self.logger.info(f'Using {n_basis_vectors} vectors with <= {max_basis_vectors} non-zeros.')

        

        if self.mode == 'custom':
            weights = tf.constant(weights, dtype='float32')
            w_init = tf.random_normal_initializer()
            zeros_init = tf.zeros_initializer()
            basis = tf.Variable(name='basis', initial_value=w_init(shape=(features, n_basis_vectors)), trainable=True, dtype='float32')
            sparse_dict = tf.Variable(name='sparse_code', initial_value=zeros_init(shape=(n_basis_vectors, units)),
            constraint=(kNonZeroes(max_basis_vectors)), trainable=True,
            dtype='float32')


            @tf.function
            def train_step_sparse(basis, sparse_dict, weights):
                with tf.GradientTape() as (tape):
                    pred = tf.matmul(basis, sparse_dict)
                    loss = tf.reduce_mean(tf.square(weights - pred)) 
                gradients = tape.gradient(loss, [basis, sparse_dict])
                optimizer.apply_gradients(zip(gradients, [basis, sparse_dict]))
                return loss

            start_time = datetime.now()
            optimizer = tf.keras.optimizers.Adam(1e-6) 
            for i in range(self.new_layer_iterations):
                loss = train_step_sparse(basis, sparse_dict, weights)
                if self.new_layer_verbose and i%100==0:
                    self.logger.info(f'Epoch {i} of basis x sparse Loss: {loss}')
            
            training_time = (datetime.now() - start_time).total_seconds()
            pred = tf.matmul(basis, sparse_dict)
            loss = tf.reduce_mean(tf.square(weights - pred))
            self.logger.info(f'Took {training_time} secs for {self.new_layer_iterations} iterations and {loss} MSE.')

        else:


            def sparse_coding_stage(Y, D, X, max_non_zeros):
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max_non_zeros)
                # Find X values for |Y-DX|.
                omp.fit(D, Y)
                X = tf.convert_to_tensor(omp.coef_, dtype=tf.float32)
                # Transpose so that code is in columns.
                X = tf.transpose(X)
                return X

            @tf.function
            def get_dk_vals(k, Y, D, X):
                # Create mask to remove column k.
                Dj = tf.ones_like(D)
                # Generate index for tensor update
                rows = tf.expand_dims(tf.range(Dj.shape[0]), axis=-1)
                cols = tf.expand_dims(tf.repeat(k, Dj.shape[0]), axis=-1)
                # Concatenate rows and cols index.
                indices = tf.concat([rows, cols], axis=-1)
                # Set to 0 column k to omit it from the calculation.
                Dj = tf.tensor_scatter_nd_update(Dj, indices=indices, updates=tf.repeat(0.0, repeats=Dj.shape[0]))
                Dj = D*Dj

                Xj = tf.ones_like(X)
                # Mask X to set to 0 all values in row k.
                rows = tf.expand_dims(tf.repeat(k, Xj.shape[1]), axis=-1)
                cols = tf.expand_dims(tf.range(Xj.shape[1]), axis=-1)
                indices = tf.concat([rows, cols], axis=-1)
                Xj = tf.tensor_scatter_nd_update(Xj, indices=indices, updates=tf.repeat(0.0, repeats=Xj.shape[1]))
                Xj = X*Xj
                Ek = Y - tf.matmul(Dj,Xj)

                xkT = tf.gather(X, indices=k)
                # Get indexes where xkT(i)!=0.
                wk = tf.cast(tf.where(xkT!=0.0), tf.int32)
                wk = tf.squeeze(wk)
                # Create Omega k matrix.
                omegak = tf.zeros(shape=[tf.shape(Y)[1], tf.shape(wk)[0]], dtype=tf.float32)

                # Set to 1.0 the elements wk(i), i.
                rows = tf.expand_dims(wk, axis=-1)
                cols = tf.expand_dims(tf.range(tf.shape(wk)[0]), axis=-1)
                indices = tf.concat([rows, cols], axis=1)
                omegak = tf.tensor_scatter_nd_update(omegak, indices=indices, updates=tf.repeat(1.0, tf.shape(wk)[0]))

                # Get values of xkT that are !=0.
                # xkR = tf.matmul(tf.expand_dims(xkT, axis=0),omegak)
                # yrk= tf.matmul(Y, omegak)
                ekr = tf.matmul(Ek, omegak)
                s, u, vT = tf.linalg.svd(ekr, full_matrices=False, compute_uv=True)
                new_dk = tf.gather(tf.transpose(u), indices=0)
                rows = tf.expand_dims(tf.range(tf.shape(D)[0]), axis=-1)
                cols = tf.expand_dims(tf.repeat(k, tf.shape(new_dk)[0]), axis=-1)
                indices = tf.concat([rows, cols], axis=1)
                return tf.transpose(new_dk)
            

            Y = tf.convert_to_tensor(weights, np.float32)

            kmeans = KMeans(n_clusters=n_basis_vectors).fit(weights.T)


            # Set centroids as initial dictionary.
            D = tf.convert_to_tensor(kmeans.cluster_centers_.T, np.float32)
            # Initialize X to use only closest centroid.
            X = np.zeros((n_basis_vectors, units), dtype=np.float32)
            X[kmeans.labels_, range(units)] = 1.0
            X = tf.convert_to_tensor(X)
            loss = tf.reduce_mean(tf.square(Y-tf.matmul(D,X)))
            self.logger.info(f'Initial Mean Squared Error (MSE) is {loss}...')

            # Process all columns in parallel.
            parallel_iterations = D.shape[-1]
            # Generate elems for map fn.
            num_cols = tf.range(D.shape[-1])
            best_loss = loss
            best_basis_D = tf.identity(D)
            best_codebook_X = tf.identity(X)
            best_it = 0
            for J in range(self.new_layer_iterations):
                X = sparse_coding_stage(Y, D, X, max_basis_vectors)
                D = tf.map_fn(lambda k: get_dk_vals(k, Y, D, X), num_cols, dtype=tf.float32, parallel_iterations=parallel_iterations)
                D = tf.transpose(D)
                loss = tf.reduce_mean(tf.square(Y-tf.matmul(D,X)))
                if loss < best_loss:
                    best_basis_D = tf.identity(D)
                    best_codebook_X = tf.identity(X)
                    best_it = J
                    best_loss = loss
                if J % 10 == 0:
                    self.logger.debug(f'Iteration {J} has {loss} MSE.')

            loss = tf.reduce_mean(tf.square(Y-tf.matmul(best_basis_D,best_codebook_X)))
            self.logger.info(f'Best solution found during iteration {best_it} has {best_loss} MAE.')
            
            basis = tf.identity(best_basis_D)
            sparse_dict = tf.identity(best_codebook_X)

        new_layer = SparseSVD(units=units, basis_vectors=n_basis_vectors, k_basis_vectors=max_basis_vectors, activation=activation,
                  name=old_layer.name + '/SparseSVD')

        new_layer(old_layer.input)
        new_layer.set_weights([basis.numpy(), sparse_dict.numpy(), bias])

        num_zeroes_sparse = tf.math.count_nonzero(sparse_dict == 0.0).numpy()
        non_zeroes_sparse = tf.math.count_nonzero(sparse_dict != 0.0).numpy()

        self.logger.debug(f'Sparse dict has {num_zeroes_sparse} zeroes and {non_zeroes_sparse} non-zeroes.')
        
        weights_before = np.sum([K.count_params(w) for w in old_layer.trainable_weights])
        weights_after = tf.size(basis) + 2* non_zeroes_sparse + tf.size(bias)

        self.logger.debug(f'Basis has {tf.size(basis)} weights, sparse dict {non_zeroes_sparse} and bias {tf.size(bias)}')

        return new_layer, new_layer.name, weights_before, weights_after



class InsertSVDConv(ModelCompression):
    __doc__ = '\n    Compression techniques that reduces the number of weights in a filter by\n    applying the filter in two steps, one horizontal and one vertical. Instead of\n    using a DxD filter, a Dx1 is used followed by a 1xD filter. Thus, reducing the\n    number of required weights. The process to find the weights of the one\n    dimensional filters is by regressing a convolutional neural network and\n    setting the output of the original filter as the target of the regression\n    model.\n    '

    def __init__(self, **kwargs):
        (super(InsertSVDConv, self).__init__)(**kwargs)
        self.target_layer_type = 'conv'

    def get_new_layer(self, old_layer):
        config = old_layer.get_config()
        filters = config['filters']
        padding = config['padding']
        strides = config['strides']
        kernel_size = config['kernel_size']
        units = filters //12

        new_layer = ConvSVD(hidden_filters=units, filters=filters, strides=strides, kernel_size=kernel_size, padding=padding,name=old_layer.name + '/SVDConv')
        new_layer._name = old_layer.name + '/SVDConv'

        new_layer(old_layer.input)
        weights_before = np.sum([K.count_params(w) for w in old_layer.trainable_weights])
        weights_after =  np.sum([K.count_params(w) for w in new_layer.trainable_weights])
        
        return new_layer, new_layer.name, weights_before, weights_after
  
class DepthwiseSeparableConvolution(ModelCompression):
    __doc__ = '\n    Compression techniques that reduces the number of weights in a filter by\n    applying the filter in two steps, one horizontal and one vertical. Instead of\n    using a DxD filter, a Dx1 is used followed by a 1xD filter. Thus, reducing the\n    number of required weights. The process to find the weights of the one\n    dimensional filters is by regressing a convolutional neural network and\n    setting the output of the original filter as the target of the regression\n    model.\n    '

    def __init__(self, **kwargs):
        (super(DepthwiseSeparableConvolution, self).__init__)(**kwargs)
        self.target_layer_type = 'conv'

    def get_new_layer(self, old_layer):
        config = old_layer.get_config()
        filters = config['filters']
        kernel_size = config['kernel_size']
        padding = config['padding']
        new_layer = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                  name=old_layer.name + '/DepthwiseSeparableLayer')

        new_layer(old_layer.input)

        weights_before = np.sum([K.count_params(w) for w in old_layer.trainable_weights])
        weights_after = np.sum([K.count_params(w) for w in new_layer.trainable_weights])

        self.logger.debug(f'Replaced layer was using {weights_before} weights.')
        self.logger.debug(f'DepthWise Separable is using {weights_after} weights.')

        return new_layer, new_layer.name, weights_before, weights_after


class FireLayerCompression(ModelCompression):
    __doc__ = '\n    Compression techniques that replaces a convolutional layer by a fire layer,\n    which consists of 1x1 and 3x3 convolutions. A 1x1 is\n    '

    def __init__(self, **kwargs):
        (super(FireLayerCompression, self).__init__)(**kwargs)
        self.target_layer_type = 'conv'

    def get_new_layer(self, old_layer):
        config = old_layer.get_config()
        filters = config['filters']
        kernel_size = config['kernel_size']
        activation = config['activation']
        strides = config['strides']
        padding = config['padding']

        squeeze_filters = (filters // 4)
        # expand1x1_filters = (filters // 2)
        # expand3x3_filters = (filters // 2)
        new_layer = FireLayer(squeeze_filters=squeeze_filters, filters=filters, kernel_size=kernel_size, activation=activation,
                  padding=padding,
                  strides=strides,
                  name=old_layer.name + '/FireLayer')

        new_layer(old_layer.input)


        weights_before = np.sum([K.count_params(w) for w in old_layer.trainable_weights])
        weights_after = np.sum([K.count_params(w) for w in new_layer.trainable_weights])

        self.logger.debug(f'Replaced layer was using {weights_before} weights.')
        self.logger.debug(f'FireLayer is using {weights_after} weights.')

        return new_layer, new_layer.name, weights_before, weights_after


class MLPCompression(ModelCompression):
    __doc__ = '\n    Compression techniques that replaces a convolutional layer by a Multi-layer\n    perceptron that learns to generate the output of each filter.\n    '

    def __init__(self, **kwargs):
        (super(MLPCompression, self).__init__)(**kwargs)
        self.target_layer_type = 'conv'

    def get_new_layer(self, old_layer):
        config = old_layer.get_config()
        filters = config['filters']
        kernel_size = config['kernel_size']
        activation = config['activation']
        padding = config['padding']
        
        kernel, bias = old_layer.get_weights()

        weights = tf.reshape(kernel, shape=[-1, filters])
        if self.percentage is None:
            if self.hidden_units is None:
                hidden_units = filters//6
                self.percentage = 100 * hidden_units / filters
            else:
                hidden_units = self.hidden_units
                self.percentage = 100 *  hidden_units / filters
        else:
            input_size, _ = weights.shape
            max_filters = (input_size * filters)//(input_size+filters)
            if isinstance(self.percentage, int) or np.issubdtype(self.percentage, np.integer):
                hidden_units = math.ceil(max_filters*(self.percentage/100))
            elif isinstance(self.percentage, float) or isinstance(self.percentage, np.floating):
                hidden_units = math.ceil(max_filters*self.percentage)
                self.percentage *= 100
            
        self.logger.debug(f'MLP SVD is being calculated for shape {weights.shape} using {hidden_units} singular values ({self.percentage}% of units).')

        try:
            s, u, v = tf.linalg.svd(weights, full_matrices=False)
        except Exception as e:
            print(e)
            with tf.device('/CPU:0'):
                s, u, v = tf.linalg.svd(weights, full_matrices=False)

        u = tf.slice(u, begin=[0, 0], size=[weights.shape[0], hidden_units])
        s = tf.slice(s, begin=[0], size=[hidden_units])

        # Transpose V as V is returned as V^T.
        v = tf.slice(tf.linalg.matrix_transpose(v), begin=[0, 0], size=[hidden_units, filters])
        n = tf.matmul(tf.linalg.diag(s), v)
        new_weights = tf.matmul(u, n)
        loss = tf.reduce_mean(tf.square(weights - new_weights))
        self.logger.info(f'New weights have MSE of {loss}.')


        new_layer = MLPConv(filters=filters, hidden_units=hidden_units, kernel_size=kernel_size, padding=padding,
                  activation=activation,
                  name=old_layer.name + '/MLPConv')
                
        new_layer(old_layer.input)

        new_layer.set_weights([u, n, bias])

        weights_before = np.sum([K.count_params(w) for w in old_layer.trainable_weights])
        weights_after = np.sum([K.count_params(w) for w in new_layer.trainable_weights])

        self.logger.debug(f'Replaced layer was using {weights_before} weights.')
        self.logger.debug(f'MLPConv is using {weights_after} weights.')

        return new_layer, new_layer.name, weights_before, weights_after



class SparseConnectionsCompression(ModelCompression):
    __doc__ = '\n    Compression technique that sparsifies the connections between input channels\n    and output channels when applying filters in a convolutional layer.\n    '

    def __init__(self, **kwargs):
        (super(SparseConnectionsCompression, self).__init__)(**kwargs)
        self.target_layer_type = 'conv'

    def get_new_layer(self, old_layer):
        config = old_layer.get_config()
        filters = config['filters']
        kernel_size = config['kernel_size']
        activation = config['activation']
        padding = config['padding']

        input_channels = old_layer.input.shape[-1]
        
        # Increase number of connections added per epoch if target will not be reached.
        if self.target_perc > self.conn_perc_per_epoch * self.tuning_epochs:
            self.logger.debug(f'Number of epochs is not enough to meet target percentage {self.target_perc} < {self.conn_perc_per_epoch * self.tuning_epochs}')
            self.conn_perc_per_epoch = self.conn_perc_per_epoch/(self.tuning_epochs-2)
            self.loger.debug(f'New target percentage is {self.conn_perc_per_epoch}')

        total_connections = filters * input_channels
        self.logger.debug(f'Using {total_connections} connections')
        connections = np.zeros(total_connections, dtype=(np.uint8))
        remaining_connections = list(range(total_connections))
        new_connections = np.random.choice(remaining_connections, (math.ceil(connections.shape[0] * self.conn_perc_per_epoch)),
          replace=False)
        connections[new_connections] = 1
        self.logger.debug(f'Starting sparse connnections with {np.sum(connections)} connections.')


        new_layer = SparseConnectionsConv2D(sparse_connections=connections, filters=filters, padding=padding,
                  activation=activation,
                  kernel_size=kernel_size,
                  name=old_layer.name + '/SparseConnectionsConv')

        self.logger.debug('Creating sparse callback.')
        cb = AddSparseConnectionsCallback(new_layer.name, target_perc=self.target_perc, conn_perc_per_epoch=self.conn_perc_per_epoch)


        if self.callbacks is None:
            self.callbacks = [cb]
        else:
            self.callbacks.append(cb)
        

        new_layer(old_layer.input)

        weights_before = np.sum([K.count_params(w) for w in old_layer.trainable_weights])

        
        if isinstance(kernel_size, int):
            channel_weights = kernel_size**2
        else:
            channel_weights = tf.reduce_prod(kernel_size)


        weights_after = weights_before - int(total_connections * (1-self.target_perc)) * channel_weights 

        self.logger.debug(f'Sparse Connections should use {weights_after} of {weights_before} weights. Final number of weights will be calculated after fine-tuning.')
       

        return new_layer, new_layer.name, weights_before, weights_after

class SparseConvolutionCompression(ModelCompression):
    __doc__ = '\n    Compression technique that performs an sparse convolution\n    '

    def __init__(self, **kwargs):
        (super(SparseConvolutionCompression, self).__init__)(**kwargs)
        self.bases = None
        self.target_layer_type = 'conv'

    def find_pqs(self, layer):
        w_init = tf.random_normal_initializer()
        identity_initializer = tf.initializers.Identity()
        weights = tf.constant(layer.get_weights()[0])
        
        sh, sw, channels, filters = weights.shape
        R = tf.Variable(name='R', initial_value=w_init(shape=(weights.shape), dtype='float32'),
          trainable=False)
        P = tf.Variable(
            name='P', initial_value=identity_initializer(shape=(channels, channels), dtype='float32'), 
            constraint=tf.keras.constraints.MaxNorm(max_value=1.0, axis=-1),
            trainable=True)
        self.logger.debug('Searching for matrix P.')


        @tf.function
        def train_step_RP(R, P, weights):
            with tf.GradientTape() as tape:
                tape.watch([R,P])
                pred = tf.einsum('hwcf,cp->hwpf',R,P)
                loss = tf.reduce_mean(tf.square(((weights - pred))))
            gradients = tape.gradient(loss, [R, P])
            optimizer.apply_gradients(zip(gradients, [R,P]))
            return loss

        optimizer = tf.keras.optimizers.Adam()
        start_time = datetime.now()
        for i in range(self.new_layer_iterations):
            loss = train_step_RP(R, P, weights)
            if self.new_layer_verbose and i % 100 == 0:
                self.logger.info(f'Epoch {i} of RxP Loss: {loss}')

        training_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f'Took {training_time} secs for {self.new_layer_iterations} iterations and {loss} MSE.')


        self.logger.debug('Searching for matrices Q and S.')
        zeroes = tf.zeros_initializer()
        S = tf.Variable(name='S', initial_value=w_init(shape=(channels, self.bases, filters),
          dtype='float32'),
          constraint = SparseWeights(1e-6),
          trainable=True)
        Q = tf.Variable(name='Q', initial_value=w_init(shape=(channels, sh, sw, self.bases),
          dtype='float32'), constraint=tf.keras.constraints.MaxNorm(max_value=1.0, axis=-1),
          trainable=True)

        for i in range(sh):
            Q[:, i, i, :].assign(tf.ones(shape=(channels, self.bases)))


        optimizer = tf.keras.optimizers.Adam(1e-5)

        @tf.function
        def train_step_SQ(S, Q, expected_value):
            SQ = tf.einsum('ikj,iuvk->uvij', S, Q)
            with tf.GradientTape() as tape:
                tape.watch([S,Q])
                SQ = tf.einsum('ikj,iuvk->uvij', S, Q)
                # Removed l2 norm cause it causes nan.
                loss = tf.reduce_mean(tf.square(expected_value - SQ)) #+ 0.001 * tf.reduce_sum(tf.abs(S)) + 0.001 * tf.reduce_sum(tf.math.sqrt(tf.reduce_sum(tf.square(weights), axis=-1)))
            gradients = tape.gradient(loss, [S, Q])
            optimizer.apply_gradients(zip(gradients, [S, Q]))     
            return loss

        

        expected_value = tf.constant(R)
        start_time = datetime.now()
        for i in range(self.new_layer_iterations_sparse):
            loss = train_step_SQ(S, Q, expected_value)
            if self.new_layer_verbose and i % 1000 == 0:
                self.logger.debug(f'Epoch {i} Loss: {loss}')
            
            

        training_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f'Took {training_time} secs for {self.new_layer_iterations_sparse} iterations and {loss} MSE.')

        self.logger.debug(f'Matrix P has shape {P.shape}')
        self.logger.debug(f'Matrix Q has shape {Q.shape}')
        self.logger.debug(f'Matrix S has shape {S.shape}')
        return P.numpy(), Q.numpy(), S.numpy()

    def get_new_layer(self, old_layer):
        config = old_layer.get_config()
        kernel, bias = old_layer.get_weights()
        s, _, channels, filters = kernel.shape
        # Previous calculation
        # self.bases = (9*channels*filters-channels**2)//(9*channels+channels*filters)
        upper_limit = (channels*filters*s**2-channels**2)/(channels*s**2+channels*filters)
        self.bases = int(upper_limit//2)

        self.logger.debug(f'Using {self.bases} bases for {channels} input channels. Upper limit is {upper_limit}')
        activation = config['activation']
        kernel_size =config['kernel_size']
        padding = config['padding']

        P, Q, S = self.find_pqs(old_layer)
        new_layer = SparseConvolution2D(kernel_size=kernel_size, filters=filters, padding=padding,
                  activation=activation,
                  bases=(self.bases),
                  name=old_layer.name + '/SparseConv2D')

        new_layer(old_layer.input)
        new_layer.set_weights([P, Q , S, bias])

        weights_before = np.sum([K.count_params(w) for w in old_layer.trainable_weights])
        num_non_zeros = tf.math.count_nonzero(S).numpy()
        weights_after = tf.size(P) + tf.size(Q) + (2 * num_non_zeros) + tf.size(bias)


        self.logger.debug(f'Replaced layer was using {weights_before} weights.')
        self.logger.debug(f'Sparse Conv2D is using {weights_after} weights. It has {num_non_zeros} non-zeroes. {tf.size(S).numpy()-num_non_zeros} were not counted.')


        return new_layer, new_layer.name, weights_before, weights_after

        

# okay decompiling CompressionTechniques.cpython-36.pyc
