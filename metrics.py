import numpy as np
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
import tensorflow as tf

def sparse_categorical_crossentropy(y_true, y_pred):
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    class_proba = y_pred[np.arange(len(y_true)), y_true]
    return -np.log(class_proba)

def categorical_crossentropy(y_true, y_pred):
    indexes = np.argmax(y_true, axis=1)
    return sparse_categorical_crossentropy(indexes, y_pred)

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

class Metric(ABC):
    def __init__(self):
        self.weights = []

    @abstractmethod
    def update_state(self, y_true, y_pred, sample_weights=None):
        pass

    @abstractmethod
    def result(self):
        pass

    def get_config(self):
        return self.__dict__.copy()
    
    def __call__(self, y_true, y_pred):
        self.update_state(y_true, y_pred)
        return self.result()
    
    def add_weights(self, name, initializer):
        self.weights.append(name)
        ...
        return

class Precision(Metric):
    def __init__(self):
        self.true_positives = tf.Variable(0.)
        self.false_positives = tf.Variable(0.)

    def update_state(self, y_true, y_pred):
        cond1 = y_true == y_pred
        cond2 = y_pred == 1
        tp_mask = tf.where(cond1 & cond2, 1., 0.)
        fp_mask = tf.where(~cond1 & cond2, 1., 0.)
        tp = tf.reduce_sum(tp_mask)
        fp = tf.reduce_sum(fp_mask)
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        tp = self.true_positives
        fp = self.false_positives
        return tp / (tp + fp)
    
    def reset_states(self):
        self.true_positives = tf.Variable(0.)
        self.false_positives = tf.Variable(0.)

class RootMeanSquaredError(Metric):
    def __init__(self):
        self.average = 0
        self.count = 0
        self.name = 'rmse'

    @staticmethod
    def _compute_rmse(y_true, y_pred):
        y_pred = tf.reshape(y_pred, y_true.shape)
        return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))

    def update_state(self, y_true, y_pred, sample_weights=None):
        rmse = self._compute_rmse(y_true, y_pred)
        self.average += rmse
        self.count += 1

    def result(self):
        return self.average / self.count
    
    def reset_states(self):
        self.average = 0
        self.count = 0

class SparseCategoricalAccuracy(Metric):
    def __init__(self):
        self.name = 'accuracy'
        self.accuracy = tf.Variable(0.0, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weights=None):
        result = tf.argmax(y_pred, axis=1)
        correct = tf.cast(tf.where(result == y_true, 1, 0), tf.float32)
        self.accuracy.assign(tf.reduce_mean(correct))
     
    def result(self):
        return self.accuracy
    
    def reset_states(self):
        self.accuracy.assign(0.0)