
import time, math

import numpy as np

import tensorflow as tf


#%% Parameters

batch_size = 128
num_classes = 10
epochs = 20

xi = 1e-6
# xi = 0.01
alpha = 1
epsilon = 2

#%% Data split parameters

# train-validation-test split
data_split = {'trainIDs': range(50000), 'valIDs': range(50000, 60000), 'testIDs': range(60000, 70000)}

# an indicator array indicatig whether a training example is labeled
# (in the training set, first 10,000 samples are labeled)
labeled = np.ones((70000, ), dtype = bool)
labeled[10000:50000, ...] = False

#%% Data load and prep

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x = np.concatenate((x_train, x_test)).reshape(70000, 784).astype('float32')
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

#%% Data generators

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



train_gen = SimpleSequence(data_split['trainIDs'], batch_size,
                          data = get_data_subset(data, data_split, 'train'))

val_gen = SimpleSequence(data_split['valIDs'], batch_size,
                          data = get_data_subset(data, data_split, 'val'))

test_gen = SimpleSequence(data_split['testIDs'], batch_size,
                          data = get_data_subset(data, data_split, 'test'))

#%% Losses

def kl_divergence(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.losses.kl_divergence(y_true, y_pred))

#%% Adverarial attacks

def virtual_L2_attack(x, y):
    """
    Compute a virtual adversarial attack as in the original VAT paper.

    Parameters
    ----------
    x : tensor
        Clean example.
    y : tensor
        Target: should be the model prediction w.r.t. x.

    Returns
    -------
    tensor
        Virtual adversarial example.

    """
    
    noise = tf.random.normal(tf.shape(x))
    noise /= tf.norm(noise, 2, axis = 1, keepdims = True)

    inp = x + noise*xi
    
    # this can greatly increase the effectiveness of a VAT attack
    # when uncommented
    y = tf.one_hot(tf.keras.backend.argmax(y), depth = num_classes)

    return L2_attack(inp, y) - inp + x

def L2_attack(x, y):
    """
    Compute a regular L2 attack.

    Parameters
    ----------
    x : tensor
        Clean example.
    y : tensor
        Target label.

    Returns
    -------
    x_adv : tensor
        Adversarial example.

    """
    
    with tf.GradientTape() as tape:

        tape.watch(x)

        pred = model(x)
        loss_value = kl_divergence(y, pred)

    grads = tf.stop_gradient(tape.gradient(loss_value, x))
    
    # clipping is important since gradients can be very small,
    # particularly with the VAT attack
    radv = grads / tf.keras.backend.clip(tf.norm(grads, 2, axis = 1, keepdims = True), 1e-6, 1e+6)
    
    x_adv = x + epsilon * radv

    return x_adv


#%% Model training

class VATModel(tf.keras.Model):
    # def __init__(self):
    #     super(GradModel, self).__init__()

    
    def compile(self, optimizer, loss, metrics = [], run_eagerly = False):
        """
        Compile the model.

        Parameters
        ----------
        optimizer : a keras optimizer
            A keras optimizer. See tf.keras.optimizers. 
        loss : TF function
            A loss function to be used for supervised and unsupervised terms.
        metrics : a list of keras metrics, optional
            Metrics to be computed for labeled and unlabeled adversarial or clean examples.
            See self.update_metrics to see how they are handled.
        run_eagerly : bool, optional
            If True, this Model's logic will not be wrapped in a tf.function;
            one thus can debug it more easily (e.g. print inside train_step).
            The default is False.

        Returns
        -------
        None.

        """
        super(VATModel, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.loss_trackers = [tf.keras.metrics.Mean(name = 'loss'),
                              tf.keras.metrics.Mean(name = 'loss_sup'),
                              tf.keras.metrics.Mean(name = 'loss_vat')]
        self.vat_metrics = metrics
        
        self._run_eagerly = run_eagerly
    
    
    def unpack_data(self, data):
        """
        Convert data from the genrator into a more convenient form for SSL.

        Parameters
        ----------
        data : tuple
            Tuple: x: images, y: labels, labeled: binary array indicating labeled images.
        
        """
        # x: images, y: labels, labeled: binary array indicating labeled images.
        # This format was chosen to fit dimensions keras considers valid.
        x, y, labeled = data
        
        n_labeled = tf.math.count_nonzero(labeled)
        n = tf.shape(x)[0]
        
        # select labeled images
        xl = x[:n_labeled, ...]
        yl = y[:n_labeled, ...]

        xul = x[n_labeled:, ...]
        
        return x, xl, yl, xul, n_labeled, n
    
    def compute_loss(self, data):
        """
        Compute total VAT loss:
            supervised + supervised adversarial + unsupervised adversarial.

        Parameters
        ----------
        data : tuple
            The output of the generator.

        Returns
        -------
        loss_value : scalar
            Total loss.
        loss_sup : scalar
            Supervised loss.
        loss_vat : scalar
            Adversarial loss (supervised + unsupervised).
        pred : tensor
            Predictions on clean examples (for computing metrics).
        pred_adv : tensor
            Predictions on adversarial examples (= 'adversarial predictions')
            (for computing metrics).

        """
        
        x, xl, yl, xul, n_labeled, n = self.unpack_data(data)

        # compute predictions on clean exaples
        pred = self(x)
        predl, predul = pred[:n_labeled, ...], pred[n_labeled:, ...]
        
        # supervised loss
        loss_sup = self.loss(yl, predl)
        
        # supervised adversarial examples
        xl_adv = L2_attack(xl, yl)
        
        # virtual adversarial examples
        predul = tf.stop_gradient(predul)
        
        xul_adv = virtual_L2_attack(xul, predul)
        
        x_adv = tf.concat((xl_adv, xul_adv), axis = 0)
        y_adv = tf.concat((yl, predul), axis = 0)
        
        pred_adv = self(x_adv)
        

        loss_vat = self.loss(y_adv, pred_adv)
        
        # total loss
        loss_value = loss_sup + alpha * loss_vat
        
        return loss_value, loss_sup, loss_vat, pred, pred_adv
    
    def update_metrics(self, loss_values, yl, pred, pred_adv):
        """
        Updates loss trackers and metrics so that they return the current moving average.

        """
        
        # update all the loss trackers with current batch loss values
        for loss_tracker, loss_value in zip(self.loss_trackers, loss_values):
            loss_tracker.update_state(loss_value)

        # separate predictions into labeled and unlabeled subsets
        # for metric computation        
        n_labeled = tf.shape(yl)[0]
        predl, predul = pred[:n_labeled, ...], pred[n_labeled:, ...]
        predl_adv, predul_adv = pred_adv[:n_labeled, ...], pred_adv[n_labeled:, ...]
        
        # for every metric type
        # sup:      metrics on the labeled subset measuring GT vs clean prediction fidelity
        # ladv:     metrics on the labeled subset measuring GT vs adversarial prediction fidelity
        # uladv:    metrics on the unlabeled subset measuring clean vs adversarial prediction fidelity
        # adv:      metrics on the entire batch measuring clean vs adversarial prediction fidelity
        for metric_type, y_true, y_pred in zip(['sup', 'ladv', 'uladv', 'adv'],
                                             [yl, yl, predul, pred],
                                             [predl, predl_adv, predul_adv, pred_adv]):
            
            for metric in self.vat_metrics:
                
                # if metric name contains the type name
                if metric_type in metric.name.split('_'):
                    metric.update_state(y_true, y_pred)
            
        return {m.name: m.result() for m in self.metrics}
    
    def train_step(self, data):
        """
        This method is called by model.fit() for every batch.
        It should compute gradients, update model parameters and metrics.

        Parameters
        ----------
        data : tuple
            Batch received from the generator.

        Returns
        -------
        metric_values : dictionary
            Current values of all metrics (including loss terms).

        """
        
        x, xl, yl, xul, n_labeled, n = self.unpack_data(data)

        # compute gradient wrt parameters
        with tf.GradientTape() as tape:
            loss_value, loss_sup, loss_vat, pred, pred_adv = self.compute_loss(data)

        grads = tape.gradient(loss_value, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))        
        
        metric_values = self.update_metrics([loss_value, loss_sup, loss_vat], yl, pred, pred_adv)

        return metric_values
        
    def test_step(self, data):
        """
        This method is called by model.fit() during the validation step
        and by model.evaluate().

        """
        
        x, xl, yl, xul, n_labeled, n = self.unpack_data(data)
        
        loss_value, loss_sup, loss_vat, pred, pred_adv = self.compute_loss(data)
        
        metric_values = self.update_metrics([loss_value, loss_sup, loss_vat], yl, pred, pred_adv)
        
        return metric_values

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.loss_trackers + self.vat_metrics

#%% Specify model architecture

inp = tf.keras.Input(shape=(784,))

x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(inp)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x)

model = VATModel(inputs=[inp], outputs=[output])
# model = tf.keras.Model(inputs=[inp], outputs=[output])

model.summary()

#%% Compile the model

# create an optimizer
opt = tf.keras.optimizers.Adadelta(learning_rate = 0.1)

# create metrics
metrics = [getattr(tf.keras.metrics, metric_class)(name = ('%s_%s' % (metric_type, metric_name)))
           for metric_type in ['sup', 'ladv', 'uladv', 'adv']
           for metric_class, metric_name in zip(['CategoricalAccuracy'], ['acc'])]


run_eagerly = False     # set to true to debug model training

model.compile(optimizer = opt, loss = kl_divergence,
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