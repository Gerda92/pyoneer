# Image processing functions

import numpy as np

#%% Functions used in the SSL method for transforming

def transform_random_noise(image, sigma):
    
    return np.clip(image + np.random.normal(scale = sigma, size = image.shape), \
                   np.min(image), np.max(image))