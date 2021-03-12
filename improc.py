# Image processing functions

import numpy as np

#%% Functions used in the SSL method for transforming

def transform_random_noise(image, sigma):
    
    return image + np.random.normal(scale = sigma, size = image.shape)