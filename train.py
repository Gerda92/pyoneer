# Model training and evaluation

from datagen import SimpleSequence
import func
import models
from omegaconf import OmegaConf
import tensorflow as tf
import numpy as np
import os
import time


# comment this line out to use gpu:
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %% Load and specify parameters

p = OmegaConf.load('params.yml')

run_eagerly = True     # set to true to debug model training

# %% Data split parameters


# train-validation-test split

# semi-supervised learning:
if p.data_split == 'cifar10_ssl_default':
    data_split = {'trainIDs': range(49000), 'valIDs': range(49000, 50000),
                  'testIDs': range(50000, 60000)}
# supervised learning:
elif p.data_split == 'cifar10_default':
    data_split = {'trainIDs': range(4000), 'valIDs': range(49000, 50000),
                  'testIDs': range(50000, 60000)}
else:
    raise Exception('Data split not found: ', p.data_split)


# an indicator array indicatig whether a training example is labeled
labeled = np.ones((60000, ), dtype = bool)

if p.data_split == 'cifar10_ssl_default':
    labeled[4000:49000, ...] = False

# %% Data load and prep

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x = np.concatenate((x_train, x_test)).astype('float32')
y = np.concatenate((y_train, y_test))

# class numbers -> one-hot representation
y = tf.keras.utils.to_categorical(y)

data = {'x': x, 'y': y, 'labeled': labeled}


def get_data_subset(data, split, subset):
    """
    Select training, validation or testing portion of the data.
    """

    return {arr: data[arr][split[subset + 'IDs']] for arr in ['x', 'y', 'labeled']}

# %% Init generators


train_gen = SimpleSequence(p, data_split['trainIDs'],
                           data=get_data_subset(data, data_split, 'train'))

val_gen = SimpleSequence(p, data_split['valIDs'],
                         data=get_data_subset(data, data_split, 'val'))

test_gen = SimpleSequence(p, data_split['testIDs'],
                          data=get_data_subset(data, data_split, 'test'))


# %% Build the model architecture

model_arch = getattr(models, 'get_' + p.arch.name)(**p.arch.params)

model_arch.summary()

# %% Compile the model

# create an optimizer
opt = getattr(tf.keras.optimizers, p.optimizer.name)(**p.optimizer.params)

# create metrics
metrics = [getattr(tf.keras.metrics, metric_class)(name=('%s_%s' % (metric_type, metric_name)))
           for metric_type in ['sup', 'usup']
           for metric_class, metric_name in zip(['CategoricalAccuracy'], ['acc'])]


model = models.SemiSupervisedConsistencyModel(inputs=[model_arch.input],
                                              outputs=[model_arch.output])
model.compile(optimizer = opt, loss = getattr(func, p.loss),
              metrics = metrics, run_eagerly = run_eagerly, p = p)

# %% Train the model

start = time.time()

history = model.fit(x = train_gen,
                    epochs = p.epochs,
                    verbose = 1,
                    validation_data = val_gen)

print('Training time: %.1f seconds.' % (time.time() - start))

# %% Evaluate the model on the test set

metric_values = model.evaluate(test_gen)

for metric_name, metric_value in zip(model.metrics_names, metric_values):
    print('%s: %.3f' % (metric_name, metric_value))
