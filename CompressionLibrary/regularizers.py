import tensorflow as tf


class L1L2SRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, weights):
        regularization = tf.constant(0.0, dtype=(tf.float32))
        if self.l1:
            regularization += self.l1 * tf.reduce_sum(tf.abs(weights))
        if self.l2:
            regularization += self.l2 * tf.reduce_sum(tf.math.sqrt(tf.reduce_sum(tf.square(weights), axis=-1)))
        return regularization

    def get_config(self):
        return {'l1':self.l1, 
         'l2':self.l2}
         