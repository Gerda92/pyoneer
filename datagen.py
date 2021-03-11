
import math

import numpy as np

import tensorflow as tf

class SimpleSequence(tf.keras.utils.Sequence):
    
    def __init__(self, IDs, batch_size, shuffle = True, data = {}):
        """
        Create a generator object.

        Parameters
        ----------
        IDs : an iterable object
            Sample IDs; useful when data needs to be loaded online.
        batch_size : batch size
        shuffle : bool, optional
            Whether to shuffle samples every epoch.
        data : dictionary
            Pre-loaded data.

        Returns
        -------
        None.

        """
        self.IDs = IDs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data
        
        self.reset()
        
    def __len__(self):
        return math.ceil(len(self.IDs) / self.batch_size)
    
    # should always provide arrays of batch_size
    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) *
            self.batch_size]
        
        if batch_idx.shape[0] < self.batch_size:
            num_to_add = self.batch_size - batch_idx.shape[0]
            batch_idx = np.concatenate((batch_idx, self.indexes[:num_to_add]), axis = 0)

        batch_x = self.data['x'][batch_idx, ...]
        batch_y = self.data['y'][batch_idx, ...]
        batch_labeled = self.data['labeled'][batch_idx]
        sortd = np.argsort(np.logical_not(batch_labeled))
        
        # return training samples, labels, indexes of labeled images
        return batch_x[sortd, ...], batch_y[sortd, ...], batch_labeled[sortd]
    
    def reset(self):
        
        # updates indexes after each epoch
        self.indexes = np.arange(len(self.IDs), dtype = int)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def on_epoch_end(self):
        
        self.reset()