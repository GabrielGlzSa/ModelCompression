import tensorflow as tf


class SparseWeights(tf.keras.constraints.Constraint):
    __doc__ = '\n    Contraint that sets to zero all weights below a threshold.\n    '

    def __init__(self, min_value=1e-5):
        super(SparseWeights, self).__init__()
        self.min_value = min_value

    def __call__(self, weights):
        below_threshold = tf.abs(weights) < self.min_value
        new_weights = tf.where(below_threshold, 0.0, weights)
        return new_weights
        
class kNonZeroes(tf.keras.constraints.Constraint):
    __doc__ = '\n    Contraint that does not set to zero the top K values per unit.\n    '

    def __init__(self, k):
        super(kNonZeroes, self).__init__()
        self.k = k

        # Added flatten layer as a workaround for reshape not working in graph mode.
        self.flatten= tf.keras.layers.Flatten()

    def __call__(self, weights):
        
        # (inputs,units) to (units,inputs)
        transposed_weights = tf.transpose(weights)

        # Get the top values and their positions per row.
        values, positions = tf.math.top_k(transposed_weights, self.k, sorted=False)

        # Flatten the values as it previously had a shape of (units, k).
        # values = tf.reshape(values, shape=-1)

        # Flatten does not work with graph mode. Workaround using Flatten layer.
        # Add batch dimension so that other dimensions can be flattened.
        values = tf.expand_dims(values, axis=0)
        values = self.flatten(values)
        # Remove batch dimension.
        values = tf.squeeze(values)

        # Flatten positions and add another dimension for the row index.
        # positions = tf.reshape(positions, shape=-1)
        # Add batch dimension so that other dimensions can be flattened.
        positions = tf.expand_dims(positions, axis=0)
        positions = self.flatten(positions)
        # Remove batch dimension.
        positions = tf.squeeze(positions)
        positions = tf.expand_dims(positions, axis=-1)

        # Generate row index for the k values.
        rows = tf.repeat(tf.range(transposed_weights.shape[0]), self.k)
        rows = tf.expand_dims(rows, axis=-1)
        positions = tf.concat([rows,positions], axis=-1)

        # Set to 0 all weights except for the top k per unit.
        weights = tf.scatter_nd(positions, values, transposed_weights.shape)
        weights = tf.transpose(weights)
        return weights