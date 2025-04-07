import tensorflow as tf
from abc import ABC, abstractmethod
from scipy.stats import norm

class Loss(ABC):
    @abstractmethod
    def forward(self, y_truye, y_pred, *args, **kwargs):
        pass
    
class MSELoss(Loss):
    @staticmethod
    def forward(y_true, y_pred, *args, **kwargs):
        return tf.reduce_mean(tf.square(y_true - y_pred))

class Softmax:
    @staticmethod
    def forward(inputs, *args, **kwargs):
        exp = tf.exp(inputs - tf.reduce_max(inputs, axis=-1, keepdims=True))
        return exp / tf.reduce_sum(exp, axis=-1, keepdims=True)

class CategoricalCrossEntropyLoss(Loss):
    @staticmethod
    def forward(y_true, y_pred, *args, **kwargs):
        # Handle both one-hot encoded and class index inputs
        if len(y_true.shape) > 1:
            # If one-hot encoded, convert to class indices
            y_true = tf.argmax(y_true, axis=-1)
        
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true_indices = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
        true_probs = tf.gather_nd(y_pred, y_true_indices)
        return -tf.reduce_mean(tf.math.log(true_probs + 1e-10))  # Added small epsilon to avoid log(0)

class EmptyActivation:
    def forward(self, inputs, *args, **kwargs):
        return inputs

class ActivationMixin:
    def set_size(self, size):
        self.size = size

    def get_size(self):
        return self.size
    
    def copy(self):
        a = self.__class__()
        a.__dict__ = self.__dict__.copy()
        if hasattr(a, 'inputs'):
            a.inputs = self.inputs.copy()
        return a
    
class ReLU(ActivationMixin):
    @staticmethod
    def forward(inputs, *args, **kwargs):
        return tf.maximum(0., inputs)
  
class Tanh(ActivationMixin):
    @staticmethod
    def forward(inputs, *args, **kwargs):
        return tf.tanh(inputs)

class LeakyReLU(ActivationMixin):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    @staticmethod
    def _forward(alpha, inputs):
        tf.maximum(alpha * inputs, inputs)

    def forward(self, inputs, *args, **kwargs):
        return self._forward(self.alpha, inputs)

class PReLU(ActivationMixin):
    def __init__(self, alpha=0.01, learning_rate=0.001):
        self.alpha = alpha
        self.learning_rate = learning_rate

    def forward(self, inputs, *args, **kwargs):
        return tf.maximum(self.alpha * inputs, inputs)
    
class GELU(ActivationMixin):
    @staticmethod
    def forward(self, inputs, *args, **kwargs):
        return inputs * norm.cdf(inputs)

class Swish(ActivationMixin):
    def __init__(self, beta=1, learning_rate = 0.001):
        self.beta = beta
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(inputs):
        # FIXME overflow
        return 1 / (1 + tf.exp(-inputs))

    def forward(self, inputs, *args, **kwargs):
        self.inputs = inputs
        self.s = self.sigmoid(self.beta * self.inputs)
        self.f_x = inputs * self.s
        return self.f_x

class ELU(ActivationMixin):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def forward(self, inputs, *args, **kwargs):
        return tf.where(inputs >= 0, inputs, self.alpha * (tf.exp(inputs) - 1))