# Image processing functions

import os

import numpy as np

from PIL import Image
from scipy.ndimage import zoom


#%% Functions used in the SSL method for transforming

def get_random_gaussian_noise(shape, sigma):
    
    return np.random.normal(scale = sigma, size = shape)
        
def get_random_shift_displacement_map(shape, max_shift):
    
    batch_size = shape[0]
    image_size = shape[-2:]
    
    # generating random 2D shift vectors:
    shift = np.random.randint(-max_shift, max_shift, size = (batch_size, 2))
    
    # computing displacement vectors:
    displacement = np.tile(np.reshape(shift, (batch_size, 2, 1, 1)), \
                           [1, 1] + list(image_size))
        
    return displacement

#%% Functions for loading and preprocessing images and segmentations

# loads a batch of JSRT images
def load_batch_JSRT(p, IDs, batchIDs):
    
    original_image_shape = (2048, 2048)
    target_image_shape = (512, 512)

    images = np.zeros([p.batch_size, 1] + list(target_image_shape))
    GT = np.zeros([p.batch_size, 6] + list(target_image_shape), dtype = bool)
       
    for idx, imID in enumerate(batchIDs):
       
        ID = IDs[imID]

        path = os.path.join(p.data_path, 'JSRT/images/%s.IMG' % ID) # path to an image
       
        fid = open(path, 'rb')
        data = np.fromfile(fid, '>u2')
        image = data.reshape(original_image_shape)

        image = (4096. - np.clip(np.round(zoom(image, 0.25)), 0, 4096))
        image = image.reshape([1] + list(image.shape))

        # rescale between -1 and 1
        image = image / 4096. * 2 - 1    
 
        masks = np.zeros([6] + list(image.shape[-2:]))
       
        for cidx, st in enumerate(['heart', 'left clavicle', 'right clavicle', 'left lung', 'right lung']):
           
            path2 = os.path.join(p.data_path, 'JSRT/masks/%s/%s.gif' % (st, ID))
       
            masks[cidx+1, :, :] = zoom(np.array(Image.open(path2)) > 0, 0.5, order = 0)
       
        # background class
        masks[0, :, :] = np.sum(masks, axis = 0) == 0
       
        # subtract heart and clavicles from lungs
        masks = nonoverlapJSRT(masks)

        assert(np.all(np.sum(masks, axis = 0) == 1))
                       
        images[idx, ...] = image
        GT[idx, ...] = masks

    return images, GT

# this ensures class masks don't overlap
def nonoverlapJSRT(masks):
   
    for idx in range(4):
        mask = masks[(idx+1):(idx+2), ...]
        masks[(idx+2):, ...][np.tile(mask, (4 - idx, 1, 1)).astype(bool)] = 0
       
    masks[0, :, :] = np.sum(masks[1:, ...], axis = 0) == 0

    return masks


#%% Visualizing images and GT

def plot_batch_sample(p, batch_x, batch_y, path, n_images = 3):
    
    batch_x = (batch_x - batch_x.min()) / (batch_x.max() - batch_x.min())
    
    # if pixel-wise labels:
    if batch_y.ndim == batch_x.ndim:
        batch_y = np.tile(np.expand_dims(np.argmax(batch_y, axis = 1), axis = 1), (1, batch_x.shape[1], 1, 1)) / p.num_classes
    
        images = [np.concatenate((batch_x[idx, ...], batch_y[idx, ...]), axis = -2) \
                                 for idx in range(n_images)]
    else:
        images = [batch_x[idx, ...] for idx in range(n_images)]
    
    images = np.concatenate(images, axis = -1)
    
    images = np.moveaxis(images, source = [0], destination = [-1])
    images = np.tile(images, (1, 1, 3 // images.shape[-1]))
    
    images = (images * 256).astype(np.uint8)

    Image.fromarray(images).save(path)