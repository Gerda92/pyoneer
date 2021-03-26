# TF functions (losses, metrics etc.)

from collections.abc import Iterable


import tensorflow as tf

import elasticdeform.tf as etf


#%% Loss functions

def kl_divergence(y_true, y_pred):
    
   return tf.keras.backend.mean(tf.keras.losses.kl_divergence(y_true, y_pred))


#%% Transform ops

# this one is merely for testing purposes:
def get_batch_transform_identity(displacement):
    """
    Returns an identity mapping. Merely for testing purposes.

    """
    return lambda x: x

# add a given noise tensor and clip so that image values are not outside the normal range:
def get_batch_transform_add_noise(noise):
    """
    Add a given noise tensor and clip the result so that image value sare not outside the normal range.

    """
    return lambda x: tf.clip_by_value(x + noise, tf.reduce_min(x), tf.reduce_max(x))

# apply any displacement map using elasticdeform library:
def get_batch_transform_displacement_map(displacement, interpolation_order = 3):
    """
    Apply a displacement map using elasticdeform library.

    """
    
    fn = lambda x: (etf.deform_grid(x[0], x[1], order = interpolation_order, axis = (1, 2)), x[1])
    
    return lambda x: tf.map_fn(fn, elems = (x, displacement))[0]

