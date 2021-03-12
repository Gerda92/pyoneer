# TF functions (losses, metrics etc.)

import tensorflow as tf

def kl_divergence(y_true, y_pred):
    
    # if there are no labeled training examples:
    if tf.math.equal(tf.size(y_true), 0):
        return 0
    else:
        return tf.keras.backend.mean(tf.keras.losses.kl_divergence(y_true, y_pred))