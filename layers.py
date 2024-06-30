# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Layer

class L1Dist(Layer):
    def call(self, input_embedding, validation_embedding):
        input_tensor = tf.convert_to_tensor(input_embedding)
        validation_tensor = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_tensor - validation_tensor)