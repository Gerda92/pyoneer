import tensorflow as tf

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization


#%% Model architectures

def get_model_large(activation, dropout):

    if activation == 'LeakyReLU':
        activation = tf.keras.layers.LeakyReLU(alpha = 0.1)
    else:
        activation = tf.keras.layers.ReLU()
        
    inp = tf.keras.Input(shape=(3, 32, 32))
    x = Conv2D(96, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3))(inp)
    x = BatchNormalization()(x)
    x = Conv2D(96, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(96, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    
    x = Dropout(dropout)(x)
    
    x = Conv2D(192, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    
    x = Dropout(dropout)(x)
    
    x = Conv2D(192, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (1, 1), activation=activation, kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (1, 1), activation=activation, kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x) 
    
    output = Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=[inp], outputs=[output])
    
    return model


#%% Model classes with custom training

class SemiSupervisedConsistencyModel(tf.keras.Model):
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
        super(SemiSupervisedConsistencyModel, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.loss_trackers = [tf.keras.metrics.Mean(name = 'loss'),
                              tf.keras.metrics.Mean(name = 'loss_sup'),
                              tf.keras.metrics.Mean(name = 'loss_usup')]
        self.extra_metrics = metrics
        
        self._run_eagerly = run_eagerly
    
        
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
        
        x, y, labeled = data
        n_labeled = tf.math.count_nonzero(labeled) / 2
        n = tf.shape(x)[0] / 2
        
        print(x.shape, y.shape)

        # compute predictions on clean exaples
        pred = self([x])
        
        yl = tf.concat((pred[:n_labeled, ...], pred[:n_labeled, ...]), axis = 0)
        predl = tf.concat((pred[:n_labeled, ...], pred[n:(n+n_labeled), ...]), axis = 0)
        pred1, pred2 = pred[:n, ...], pred[n:, ...]
        
        # supervised loss
        loss_sup = self.loss(yl, predl)       

        loss_usup = (self.loss(pred1, pred2) + self.loss(pred2, pred1)) / 2
        
        # total loss
        loss_value = loss_sup + loss_usup
        
        return loss_value, loss_sup, loss_usup, (yl, predl), (pred1, pred2)
    
    def update_metrics(self, data, loss_values, pair_sup, pair_usup):
        """
        Updates loss trackers and metrics so that they return the current moving average.

        """

        # update all the loss trackers with current batch loss values
        for loss_tracker, loss_value in zip(self.loss_trackers, loss_values):
            loss_tracker.update_state(loss_value)

        # obtain prediction - target pairs       
        yl, predl = pair_sup
        pred1, pred2 = pair_usup
        
        # for every metric type
        # sup:      metrics on the labeled subset measuring GT vs clean prediction fidelity
        # usup:     metrics on the entire batch measuring consistency
        for metric_type, y_true, y_pred in zip(['sup', 'usup'],
                                             [yl, pred1],
                                             [predl, pred2]):
            
            for metric in self.extra_metrics:
                
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

        # compute gradient wrt parameters
        with tf.GradientTape() as tape:
            loss_value, loss_sup, loss_usup, pair_sup, pair_usup = self.compute_loss(data)

        grads = tape.gradient(loss_value, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))        
        
        metric_values = self.update_metrics([loss_value, loss_sup, loss_usup], pair_sup, pair_usup)

        return metric_values
        
    def test_step(self, data):
        """
        This method is called by model.fit() during the validation step
        and by model.evaluate().

        """

        loss_value, loss_sup, loss_usup, pair_sup, pair_usup = self.compute_loss(data)
        
        metric_values = self.update_metrics([loss_value, loss_sup, loss_usup], pair_sup, pair_usup)
        
        return metric_values

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.loss_trackers + self.extra_metrics