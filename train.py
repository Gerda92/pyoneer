
import time

import numpy as np

import tensorflow as tf

import models
import func
from datagen import SimpleSequence

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%% Parameters

batch_size = 8
num_classes = 10
epochs = 20

alpha = 1

activation = 'LeakyReLU'
dropout = 0.5

#%% Data split parameters

# train-validation-test split
data_split = {'trainIDs': range(49000), 'valIDs': range(49000, 50000), 'testIDs': range(50000, 60000)}

# an indicator array indicatig whether a training example is labeled
# (in the training set, first 10,000 samples are labeled)
labeled = np.ones((60000, ), dtype = bool)
labeled[4000:49000, ...] = False

#%% Data load and prep

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x = np.concatenate((x_train, x_test)).astype('float32')
y = np.concatenate((y_train, y_test))

# normalize images
x /= 255

# class numbers -> one-hot representation
y = tf.keras.utils.to_categorical(y)

data = {'x': x, 'y': y, 'labeled': labeled}

def get_data_subset(data, split, subset):
    """
    Select training, validation or testing portion of the data.
    """
    
    return {arr: data[arr][split[subset + 'IDs']] for arr in ['x', 'y', 'labeled']}

#%% Init generators

train_gen = SimpleSequence(data_split['trainIDs'], batch_size,
                          data = get_data_subset(data, data_split, 'train'))

val_gen = SimpleSequence(data_split['valIDs'], batch_size,
                          data = get_data_subset(data, data_split, 'val'))

test_gen = SimpleSequence(data_split['testIDs'], batch_size,
                          data = get_data_subset(data, data_split, 'test'))

#%% Losses

def kl_divergence(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.losses.kl_divergence(y_true, y_pred))

#%% Specify model architecture

inp = tf.keras.Input(shape=(3, 32, 32))

x = tf.keras.layers.Flatten()(inp)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x)

# model = models.get_model_large(activation, dropout)

# model = models.SemiSupervisedConsistencyModel(inputs=[model.input], outputs=[model.output])
model = models.SemiSupervisedConsistencyModel(inputs=[inp], outputs=[output])

model.summary()

#%% Compile the model

# create an optimizer
opt = tf.keras.optimizers.Adadelta(learning_rate = 0.1)

# create metrics
metrics = [getattr(tf.keras.metrics, metric_class)(name = ('%s_%s' % (metric_type, metric_name)))
           for metric_type in ['usup']
           for metric_class, metric_name in zip(['CategoricalAccuracy'], ['acc'])]


run_eagerly = True     # set to true to debug model training

model.compile(optimizer = opt, loss = func.kl_divergence,
    metrics = metrics, run_eagerly = run_eagerly)
              
#%% Train the model
              
start = time.time()  
  
history = model.fit(x = train_gen,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = val_gen)

print('Training time: %.1f seconds.' % (time.time() - start))

#%% Evaluate the model on the test set

metric_values = model.evaluate(test_gen)

for metric_name, metric_value in zip(model.metrics_names, metric_values):
    print('%s: %.3f' % (metric_name, metric_value))