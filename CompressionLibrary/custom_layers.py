import tensorflow as tf
from keras import backend


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
