import tensorflow as tf
from CompressionLibrary.custom_constraints import kNonZeroes, SparseWeights
from CompressionLibrary.regularizers import L1L2SRegularizer

@tf.keras.utils.register_keras_serializable()
class DenseSVD(tf.keras.layers.Layer):

    def __init__(self, units, hidden_units, activation='relu', **kwargs):
        (super(DenseSVD, self).__init__)(**kwargs)
        self.units = units
        self.hidden_units = hidden_units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        _, features = input_shape
        w_init = tf.random_normal_initializer()
        self.u = tf.Variable(name='u', initial_value=w_init(shape=[features, self.hidden_units], dtype='float32'))
        self.n = tf.Variable(name='n', initial_value=w_init(shape=[self.hidden_units, self.units], dtype='float32'),
          trainable=True)
        b_init = tf.zeros_initializer()
        self.bias0 = tf.Variable(name='bias0', initial_value=b_init(shape=[self.units],dtype='float32'),
          trainable=True)

    def get_config(self):
        config = super(DenseSVD, self).get_config().copy()
        config.update({'hidden_units': self.hidden_units, 'units':self.units})
        return config

    def call(self, inputs):
        x = tf.matmul(inputs, self.u)
        x = tf.matmul(x, self.n)
        return self.activation(x+self.bias0)

@tf.keras.utils.register_keras_serializable()
class SparseSVD(tf.keras.layers.Layer):

    def __init__(self, units, basis_vectors, k_basis_vectors, activation='relu', **kwargs):
        (super(SparseSVD, self).__init__)(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.units = units
        self.basis_vectors = basis_vectors
        self.k_basis_vectors = k_basis_vectors

    def build(self, input_shape):
        _, features = input_shape
        w_init = tf.random_normal_initializer()
        zeros_init = tf.zeros_initializer()
        self.basis = tf.Variable(name='basis', initial_value=w_init(shape=(features, self.basis_vectors)), dtype='float32')
        self.sparse_dict = tf.Variable(name='sparse_code', initial_value=zeros_init(shape=(self.basis_vectors, self.units)),
          constraint=(kNonZeroes(self.k_basis_vectors)),
          dtype='float32')
        
        self.bias0 = tf.Variable(name='bias0', initial_value=zeros_init(shape=self.units ,dtype='float32'),
          trainable=True)

    def get_config(self):
        config = super(SparseSVD, self).get_config().copy()
        config.update({'units':self.units, 'basis_vectors': self.basis_vectors, 'k_basis_vectors': self.k_basis_vectors})
        return config

    def call(self, inputs):
        x = tf.matmul(inputs, self.basis)
        x = tf.matmul(x, self.sparse_dict)
        return self.activation(x + self.bias0)

@tf.keras.utils.register_keras_serializable()
class ConvSVD(tf.keras.layers.Layer):

    def __init__(self, hidden_filters, filters, kernel_size, strides, padding='valid', activation='relu', **kwargs):
        (super(ConvSVD, self).__init__)(**kwargs)
        self.hidden_filters = hidden_filters
        self.filters = filters
        self.padding = padding.upper()
        if isinstance(strides, int):
            self.strides = list(strides, strides)
        else:
            if isinstance(strides, tuple) or isinstance(strides, list):
                self.strides = strides
        if isinstance(kernel_size, int):
            kernel_size = list(kernel_size, kernel_size)
 
        self.kernel_size = kernel_size
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        _,_, _, channels = input_shape
        w_init = tf.random_normal_initializer()
        zeros_init = tf.zeros_initializer()
        self.kernel0 =  tf.Variable(name='basis', initial_value=w_init(shape=[self.kernel_size[0], self.kernel_size[0],channels, self.hidden_filters]), dtype='float32', trainable=True)
        self.kernel1 =  tf.Variable(name='basis', initial_value=w_init(shape=[self.kernel_size[0], self.kernel_size[0],self.hidden_filters, self.filters]), dtype='float32', trainable=True)

        self.bias = tf.Variable(name='bias', initial_value=zeros_init(shape=self.filters ,dtype='float32'),
          trainable=True)

    def get_config(self):
        config = super(ConvSVD, self).get_config().copy()
        config.update({'hidden_filters': self.hidden_filters, 'filters': self.filters, 'kernel_size':self.kernel_size, 'strides':self.strides, 'padding':self.padding,
        'activation':self.activation})
        return config

    def call(self, inputs):
        x = tf.nn.conv2d(input=inputs, filters=self.kernel0, strides=1, padding='SAME')
        x = tf.nn.conv2d(input=x, filters=self.kernel1, strides=self.strides, padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        return self.activation(x)

@tf.keras.utils.register_keras_serializable()
class FireLayer(tf.keras.layers.Layer):

    def __init__(self, squeeze_filters, filters, kernel_size=(3,3), strides=1, activation='relu', padding='same',**kwargs):
        (super(FireLayer, self).__init__)(**kwargs)
        self.filters = filters 
        self.squeeze_filters = squeeze_filters
        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            if isinstance(strides, tuple) or isinstance(strides, list):
                self.strides = strides
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.padding = padding.upper()
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        _, _, _, channels = input_shape
        w_init = tf.random_normal_initializer()
        zeros_init = tf.zeros_initializer()

        self.kernel_squeeze = tf.Variable(name='kernel_squeeze', initial_value=w_init(shape=(1,1, channels, self.squeeze_filters) , dtype='float32'))
        self.kernel_expand1x1 = tf.Variable(name='kernel_squeeze', initial_value=w_init(shape=(1,1, self.squeeze_filters, self.filters//2) , dtype='float32'))
        self.kernel_expand3x3 = tf.Variable(name='kernel_squeeze', initial_value=w_init(shape=(self.kernel_size[0],self.kernel_size[1], self.squeeze_filters, self.filters//2) , dtype='float32'))
        self.bias = tf.Variable(name='bias', initial_value=zeros_init(shape=self.filters ,dtype='float32'), trainable=True)

    def get_config(self):
        config = super(FireLayer, self).get_config().copy()
        config.update({'squeeze_filters': self.squeeze_filters, 'kernel_size': self.kernel_size, 'strides': self.strides, 'activation':self.activation, 'padding': self.padding, 'filters': self.filters})
        return config

    def call(self, inputs):
        x = tf.nn.conv2d(input=inputs, filters=self.kernel_squeeze, strides=self.strides, padding='SAME')
        o1x1 = tf.nn.conv2d(input=x, filters=self.kernel_expand1x1, strides=self.strides, padding='VALID')
        o3x3 = tf.nn.conv2d(input=x, filters=self.kernel_expand3x3, strides=self.strides, padding=self.padding)
        if o1x1.shape != o3x3.shape:
            o1x1 = tf.keras.layers.Cropping2D(cropping=2)(o1x1)
        x = tf.keras.layers.concatenate([o1x1, o3x3], axis=3)
        x = tf.nn.bias_add(x, self.bias)
        return self.activation(x)

@tf.keras.utils.register_keras_serializable()
class MLPConv(tf.keras.layers.Layer):

    def __init__(self, filters, hidden_units, kernel_size, strides=1, padding='VALID', activation='relu', *args, **kwargs):
        (super(MLPConv, self).__init__)(*args, **kwargs)
        self.kernel_size = kernel_size
        self.hidden_units = hidden_units
        self.filters = filters
        if isinstance(strides, int):
            self.strides = (
             strides, strides)
        else:
            if isinstance(strides, tuple) or isinstance(strides, list):
                self.strides = strides
        self.activation = tf.keras.activations.get(activation)
        self.padding = padding.upper()

    def get_config(self):
        config = super().get_config().copy()
        config.update({'filters':self.filters, 'hidden_units':self.hidden_units,
        'kernel_size':self.kernel_size, 'strides': self.strides, 'padding': self.padding, 'activation': self.activation})
        return config

    def build(self, input_shape):
        _, _, _, channels = input_shape
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        self.w_0 = tf.Variable(name='kernel0', initial_value=w_init(shape=(tf.reduce_prod(self.kernel_size) * channels, self.hidden_units), dtype='float32'),
          trainable=True)
        self.w_1 = tf.Variable(name='kernel1', initial_value=w_init(shape=(self.hidden_units, self.filters), dtype='float32'),
          trainable=True)
        self.bias = tf.Variable(name='bias', initial_value=b_init(shape=(self.filters,),
          dtype='float32'),
          trainable=True)
        
    def call(self, inputs):
        fh, fw = self.kernel_size
        stride_v, stride_h = self.strides
        patches = tf.image.extract_patches(images=inputs, sizes=[
         1, fh, fw, 1],
          strides=[
         1, stride_v, stride_h, 1],
          rates=[
         1, 1, 1, 1],
          padding=self.padding)
        output = self.activation(tf.matmul(patches, self.w_0))
        output = tf.matmul(output, self.w_1)
        output = tf.nn.bias_add(output, self.bias)
        return self.activation(output)

@tf.keras.utils.register_keras_serializable()
class SparseConnectionsConv2D(tf.keras.layers.Conv2D):

    def __init__(self, sparse_connections, *args, **kwargs):
        (super(SparseConnectionsConv2D, self).__init__)(*args, **kwargs)
        self.sparse_connections = sparse_connections

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

    def get_config(self):
        config = super(SparseConnectionsConv2D, self).get_config().copy()
        config.update({'sparse_connections': self.sparse_connections})
        return config
    
    def build(self, input_shape):
        _, _, _, channels = input_shape
        sh, sw = self.kernel_size

        zeroes_init = tf.zeros_initializer()
        w_init = tf.random_normal_initializer()
        self.sparse_kernel = tf.Variable(w_init(shape=(sh,sw, channels, self.filters), dtype=(tf.float32)), name='sparse_kernel', trainable=True)
        self.bias = tf.Variable(
            name='bias', initial_value=zeroes_init(shape=(self.filters),dtype='float32'),
            trainable=True)


        self.set_connections(self.sparse_connections)

    def set_connections(self, connections):
        self.sparse_connections = connections
        self.conn_tensor = tf.constant(self.sparse_connections, dtype='float32')
        self.conn_tensor = tf.reshape((self.conn_tensor), shape=[self.sparse_kernel.shape[(-2)],self.filters])
        self.conn_tensor = tf.repeat(tf.expand_dims(self.conn_tensor, axis=0), self.kernel_size[0], axis=0)
        self.conn_tensor = tf.repeat(tf.expand_dims(self.conn_tensor, axis=0), self.kernel_size[1], axis=0)

    def get_connections(self):
        return self.sparse_connections

    def call(self, inputs):
        kernel = tf.math.multiply(self.conn_tensor, self.sparse_kernel)
        out = self.convolution_op(inputs, kernel)
        out = tf.nn.bias_add(out, self.bias)
        out = tf.nn.relu(out)
        return out

@tf.keras.utils.register_keras_serializable()
class SparseConvolution2D(tf.keras.layers.Layer):

    def __init__(self, kernel_size, filters, bases, padding='valid', strides=1, activation=None, *args, **kwargs):
        (super(SparseConvolution2D, self).__init__)(*args, **kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.activation = tf.keras.activations.get(activation)
        self.bases = bases
        self.padding = padding.upper()
        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            if isinstance(strides, tuple) or isinstance(strides, list):
                self.strides = strides

    def build(self, input_shape):
        _, _, _, channels = input_shape
        sh, sw = self.kernel_size
        identity_initializer = tf.initializers.Identity()
        
        zeroes = tf.zeros_initializer()
        self.P = tf.Variable(
            name='P', initial_value=identity_initializer(shape=(channels, channels), dtype='float32'),
            constraint=tf.keras.constraints.MaxNorm(max_value=channels, axis=-1),
            trainable=True)

        self.Q = tf.Variable(
            name='Q', initial_value=zeroes(shape=(channels, sh, sw, self.bases),dtype='float32'),
            constraint=tf.keras.constraints.MaxNorm(max_value=self.bases, axis=-1),
            trainable=True)
        

        self.S = self.add_weight(name='S', shape=(channels, self.bases, self.filters),
          dtype='float32',
          trainable=True,
          constraint=SparseWeights(),
          regularizer=L1L2SRegularizer(l1=0.1, l2=0.1))

        self.bias = tf.Variable(
            name='bias', initial_value=zeroes(shape=(self.filters),dtype='float32'),
            trainable=True)

    def get_config(self):
        config = super(SparseConvolution2D, self).get_config()
        config.update({'filters':self.filters, 'bases':self.bases,'kernel_size':self.kernel_size, 'padding':self.padding, 'activation':self.activation, 'strides':self.strides})
        return config

    def call(self, inputs):
        J = tf.matmul(inputs, self.P)

        # Move channels to the first dimension so that it be split first in the map.
        J = tf.transpose(J, perm=[3,0,1,2])
        # Add a new dimension so that channel is 1.
        J = tf.expand_dims(J, axis=-1)

        # Add a new dummy channel dimension of size 1 so that bases become number of filters.
        filters = tf.expand_dims(self.Q, axis=-2)

        tau = tf.vectorized_map(lambda tensors: tf.nn.conv2d(input=tensors[0], filters=tensors[1], strides=self.strides, padding=self.padding) , elems=(J, filters))

        # Batch is second as channel was swapped to the first dimension. channels(c), bases(q), batch(b), height(h), width(w), filters(f).
        O = tf.einsum('cqf,cbhwq->bhwf', self.S, tau)
        O = tf.nn.bias_add(O, self.bias)
        O = tf.nn.relu(O)
        return O 
    
@tf.keras.utils.register_keras_serializable()
class ROIEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_bins, *args, **kwargs):
        super(ROIEmbedding, self).__init__(*args, **kwargs)
        self.n_bins = tf.constant(n_bins)
        self.embedding_size = tf.math.reduce_sum(
            tf.math.reduce_prod(n_bins, axis=1))

    def build(self, input_shape):

        super().build(input_shape)

    @staticmethod
    def _pool_roi(feature_map, pool_shape):
        pooled_height = pool_shape[0]
        pooled_width = pool_shape[1]
        # Compute region of interest
        region_height, region_width = tf.shape(
            feature_map)[0], tf.shape(feature_map)[1]

        # Divide the region into non-overlapping areas
        h_step = tf.cast(region_height / pooled_height, tf.int32)
        w_step = tf.cast(region_width / pooled_width, tf.int32)

        areas = tf.TensorArray(tf.int32, size=pooled_height*pooled_width)

        for i in tf.range(pooled_height):
            for j in tf.range(pooled_width):
                areas = areas.write(i*pooled_height+j, [i*h_step,
                                                        j*w_step,
                                                        (i+1)*h_step if i +
                                                        1 < pooled_height else region_height,
                                                        (j+1)*w_step if j+1 < pooled_width else region_width])

        areas = areas.stack()

        def fn(x): return tf.math.reduce_max(
            feature_map[x[0]:x[2], x[1]:x[3], :], axis=[0, 1])

        pooled_features = tf.map_fn(fn, areas, fn_output_signature =tf.float32)

        return tf.reshape(pooled_features, shape=[-1])

    @staticmethod
    def _pool_rois(feature_map, n_bins):
        """
        Applies region of interest to a single image
        """
        i0 = tf.constant(1)
        m = ROIEmbedding._pool_roi(feature_map, n_bins[0])
        def cond(i, m): return tf.less(i, n_bins.shape[0])
        def body(i, m): return [
            i+1, tf.concat([m, ROIEmbedding._pool_roi(feature_map, n_bins[i])], axis=0)]
        n, m = tf.while_loop(cond, body, loop_vars=[i0, m])
        # outputs = tf.map_fn(lambda bin_size: ROIEmbedding._pool_roi(feature_map, bin_size[0], bin_size[1]), n_bins)
        return m

    @tf.function
    def call(self, x):

        def pool_image(x):
            return ROIEmbedding._pool_rois(x, tf.constant(self.n_bins))

        pooled_areas = tf.map_fn(pool_image, x)

        # Reshape to [None, embedding_size] so that autograph knows the size of the last dimension.
        return tf.reshape(pooled_areas, shape=[-1, self.embedding_size*x.shape[-1]])

@tf.keras.utils.register_keras_serializable()
class ROIEmbedding1D(tf.keras.layers.Layer):
    def __init__(self, n_bins, *args, **kwargs):
        super(ROIEmbedding1D, self).__init__(*args, **kwargs)
        self.n_bins = tf.constant(n_bins)
        self.embedding_size = tf.math.reduce_sum(self.n_bins)

    def build(self, input_shape):

        super().build(input_shape)

    @staticmethod
    def _pool_row_n_bin_size(feature_map, pool_width):

        # Compute region of interest
        region_width = tf.shape(feature_map)[0]

        # Divide the region into non-overlapping areas
        w_step = tf.cast(region_width / pool_width, tf.int32)

        areas = tf.TensorArray(tf.int32, size=pool_width)

        for i in tf.range(pool_width):
            areas = areas.write(
                i, [i*w_step, (i+1)*w_step if i+1 < pool_width else region_width])

        areas = areas.stack()

        def fn(x): return tf.math.reduce_max(feature_map[x[0]:x[1]])

        pooled_features = tf.map_fn(fn, areas, fn_output_signature =tf.float32)
        return pooled_features

    @staticmethod
    def _pool_row(feature_map, n_bins):
        """
        Applies region of interest to a single image
        """
        i0 = tf.constant(1)
        m = ROIEmbedding1D._pool_row_n_bin_size(feature_map, n_bins[0])
        def cond(i, m): return tf.less(i, n_bins.shape[0])
        def body(i, m): return [
            i+1, tf.concat([m, ROIEmbedding1D._pool_row_n_bin_size(feature_map, n_bins[i])], axis=0)]
        n, m = tf.while_loop(cond, body, loop_vars=[i0, m])
        return m

    @tf.function
    def call(self, x):

        def pool_row(x):
            return ROIEmbedding1D._pool_row(x, self.n_bins)

        pooled_areas = tf.map_fn(pool_row, x)

        # Reshape to [None, embedding_size] so that autograph knows the size of the last dimension.
        return tf.reshape(pooled_areas, shape=[-1, self.embedding_size])
