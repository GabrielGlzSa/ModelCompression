import tensorflow as tf
from keras import backend

class ROIEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_bins, *args, **kwargs):
        super(ROIEmbedding, self).__init__(*args, **kwargs)
        self.n_bins = tf.constant(n_bins)
        self.embedding_size = tf.math.reduce_sum(tf.math.reduce_prod(n_bins, axis=1))

    def build(self, input_shape):

      super().build(input_shape)

    @staticmethod 
    def _pool_roi(feature_map, pool_shape):
      pooled_height = pool_shape[0]
      pooled_width = pool_shape[1]
      # Compute region of interest
      region_height, region_width = tf.shape(feature_map)[0], tf.shape(feature_map)[1]
 
      # Divide the region into non-overlapping areas
      h_step = tf.cast(region_height / pooled_height, tf.int32)
      w_step = tf.cast(region_width / pooled_width, tf.int32)

      areas = tf.TensorArray(tf.int32, size=pooled_height*pooled_width)

      for i in tf.range(pooled_height):
        for j in tf.range(pooled_width):
          areas = areas.write(i*pooled_height+j, [i*h_step, 
                                                      j*w_step, 
                                                      (i+1)*h_step if i+1<pooled_height else region_height, 
                                                      (j+1)*w_step if j+1<pooled_width else region_width])

      
      areas = areas.stack()
      
      fn = lambda x: tf.math.reduce_max(feature_map[x[0]:x[2], x[1]:x[3], :], axis=[0,1])

      pooled_features = tf.map_fn(fn, areas, dtype=tf.float32)

      return tf.reshape(pooled_features, shape=[-1])

    @staticmethod
    def _pool_rois(feature_map, n_bins):
      """
      Applies region of interest to a single image
      """
      i0 = tf.constant(1)
      m = ROIEmbedding._pool_roi(feature_map, n_bins[0])
      cond = lambda i, m: tf.less(i, n_bins.shape[0])
      body = lambda i, m: [i+1, tf.concat([m , ROIEmbedding._pool_roi(feature_map, n_bins[i])], axis=0)]
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
          areas = areas.write(i, [i*w_step,(i+1)*w_step if i+1<pool_width else region_width])

      
      areas = areas.stack()
      
      fn = lambda x: tf.math.reduce_max(feature_map[x[0]:x[1]])

      pooled_features = tf.map_fn(fn, areas, dtype=tf.float32)
      return pooled_features

    @staticmethod
    def _pool_row(feature_map, n_bins):
      """
      Applies region of interest to a single image
      """
      i0 = tf.constant(1)
      m = ROIEmbedding1D._pool_row_n_bin_size(feature_map, n_bins[0])
      cond = lambda i, m: tf.less(i, n_bins.shape[0])
      body = lambda i, m: [i+1, tf.concat([m , ROIEmbedding1D._pool_row_n_bin_size(feature_map, n_bins[i])], axis=0)]
      n, m = tf.while_loop(cond, body, loop_vars=[i0, m])
      return m

    @tf.function
    def call(self, x):

      def pool_row(x):
        return ROIEmbedding1D._pool_row(x, self.n_bins)

      pooled_areas = tf.map_fn(pool_row, x)
      
      # Reshape to [None, embedding_size] so that autograph knows the size of the last dimension.
      return tf.reshape(pooled_areas, shape=[-1, self.embedding_size])

class MLPConvV1(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation=None, *args, **kwargs):
        super(MLPConvV1, self).__init__(*args, **kwargs)
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

    @tf.function
    def call(self, inputs):
        batch, h, w, channels = inputs.shape

        def calculate_output(x, y, img, kernel_size, w0, b0, w1, b1, activation):
            batch, h, w, channels = img.shape
            fh = kernel_size[0]
            fw = kernel_size[1]
            dh = fh // 2
            dw = fw // 2
            return activation(
                tf.matmul(
                    activation(
                        tf.matmul(tf.reshape(img[:, x - dh:x + dh + 1, y - dw:y + dw + 1, :],
                                             shape=[-1, fh * fw * channels]),
                                  w0) + b0),
                    w1) + b1)

        func = lambda position: calculate_output(position[0], position[1],
                                                 inputs,
                                                 self.kernel_size,
                                                 self.w_0, self.b_0,
                                                 self.w_1, self.b_1,
                                                 self.activation)

        output = tf.vectorized_map(func, elems=self.indexes)
        output = tf.reshape(output, shape=[h - 2, w - 2,-1, self.filters])
        output = tf.transpose(output, perm=[2, 0, 1, 3])
        return output