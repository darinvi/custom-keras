from functools import reduce
from activations import *
from metrics import *
import pickle as pkl
import tensorflow as tf
from abc import ABC, abstractmethod

tf.random.set_seed(42)
tf.config.run_functions_eagerly(True)

ACTIVATIONS = {
    'relu': ReLU,
    'softmax': Softmax,
    'tanh': Tanh,
    'lrelu': LeakyReLU,
    'prelu': PReLU,
    'gelu': GELU,
    'swish': Swish,
    'elu': ELU
}

class Input:
    def __init__(self, shape, name=None):
        self.shape = shape
        if name is not None:
            self.name = name
        self.size = shape

    def forward(self, X, *args, **kwargs):
        return X
    
    def backward(self, dvalues, *args, **kwargs):
        return dvalues

    def get_size(self):
        return self.size
    
    def copy(self):
        return Input(shape=self.shape)
    
    def save(self, *args):
        pass

    @staticmethod
    def load(name, l, i):
        return l

class Flatten:
    def __init__(self, input_shape=None, name=None):
        if input_shape:
            self.set_size(input_shape)
        if name is not None:
            self.name = name

    def forward(self, X, *args, **kwargs):
        return tf.reshape(X, (-1, self.size))        
    
    def backward(self, dvalues):
        return dvalues

    def set_size(self, size):
        if isinstance(size, tuple):
            size = reduce(lambda curr, prev: curr * prev, size, 1)
        self.size = size

    def get_size(self):
        return self.size

    def copy(self):
        return Flatten(input_shape=self.shape)

class Concat:
    def forward(self, arrays, *args, **kwargs):
        self.arr_shapes = [a.shape for a in arrays]
        return tf.concat([*arrays], axis=1)

class Normalization:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = tf.cast(tf.math.reduce_mean(X, axis=0), tf.float32)
        self.std = tf.cast(tf.math.reduce_std(X, axis=0), tf.float32)
        return self

    def forward(self, X, *args, **kwargs):
        return (X - self.mean) / self.std
    
    def set_size(self, size):
        self.size = size

    def get_size(self):
        return self.size
    
    def save(self, name, *args):
        name, suffix = name.split('.')
        with open(name+"_norm_std"+'.'+suffix, "wb") as file:
            pkl.dump(self.std, file)        
        with open(name+"_norm_mean"+'.'+suffix, "wb") as file:
            pkl.dump(self.mean, file)        

    @staticmethod
    def load(name, *args):
        name, suffix = name.split('.')
        with open(name+'_norm_std'+'.'+suffix, "rb") as file:
            std = pkl.load(file)
        with open(name+'_norm_mean'+'.'+suffix, "rb") as file:
            mean = pkl.load(file)
        norm = Normalization()
        norm.std = std
        norm.mean = mean
        return norm

class Layer(ABC):
    def forward(self, X, *args, **kwargs):
        return self.call(X, *args, **kwargs)
    
    def __call__(self, X):
        return self.call(X)

    def set_size(self, size):
        self.size = size
        self.build(input_shape=size)
    
    def get_config(self):
        return self.__dict__.copy()

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    @abstractmethod
    def call(self, X):
        pass

    @abstractmethod
    def get_size(self):
        pass

    def _init_activation(self, activation):
        if activation is None:
            self.activation = EmptyActivation() 
        elif isinstance(activation, str):
            act = ACTIVATIONS.get(activation)
            if not act:
                raise Exception(f"Invalid activation {activation}")
            self.activation = act()
        else:
            self.activation = activation

class Dense(Layer):
    def __init__(self, n_neurons, activation=None, kernel_initializer='glorot_normal', trainable=True, kernel_regularizer=None, kernel_constraint=None):
        self._init_activation(activation)
        self._init_regularizer(kernel_regularizer)
        self.n_neurons = n_neurons
        self.biases = tf.Variable(tf.zeros((1, self.n_neurons)))
        self.kernel_initializer = kernel_initializer
        self.trainable = trainable
        self.kernel_constraint=kernel_constraint

    def _init_regularizer(self, kernel_regularizer):
        if kernel_regularizer is None:
            self.kernel_regularizer = empty_regularizer
        else:
            if not hasattr(kernel_regularizer, '__call__'):
                raise Exception("Should be callable")
            self.kernel_regularizer = kernel_regularizer

    def get_size(self):
        return self.n_neurons

    def build(self, input_shape):
        if isinstance(self.kernel_initializer, VarianceScaling):
            self.weights = self.kernel_initializer.get_weights(input_shape, self.n_neurons)
            return
        
        if self.kernel_initializer not in ('glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'):
            raise Exception(f"Invalid kernel {self.kernel_initializer}")

        # FIXME use VarianceScaling
        fan_avg = (input_shape + self.n_neurons) / 2
        if not self.kernel_initializer:
            self.weights = 0.1 * tf.Variable(tf.random.normal((input_shape, self.n_neurons), mean=0, stddev=1))
            return
        
        elif 'glorot' in self.kernel_initializer:
            if 'uniform' in self.kernel_initializer:
                r = tf.sqrt(3/fan_avg)
                self.weights = tf.Variable(tf.random.uniform((input_shape, self.n_neurons), minval=-r, maxval=r))
            else:
                std = 1 / fan_avg
                self.weights = tf.Variable(tf.random.normal((input_shape, self.n_neurons), mean=0, stddev=std))
        
        elif 'he' in self.kernel_initializer:
            if 'uniform' in self.kernel_initializer:
                r = tf.sqrt(3/input_shape)
                self.weights = tf.Variable(tf.random.uniform((input_shape, self.n_neurons), minval=-r, maxval=r))

            else:
                std = 2 / input_shape
                self.weights = tf.Variable(tf.random.normal((input_shape, self.n_neurons), mean=0, stddev=std))

    def call(self, inputs, *args, **kwargs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        output = tf.linalg.matmul(inputs, self.weights) + self.biases
        return self.activation.forward(output)
    
    def get_weights(self):
        return self.weights, self.biases

    def copy(self):
        d = Dense(n_neurons=self.n_neurons)
        d.__dict__ = self.__dict__.copy()
        d.weights = self.weights.copy()
        d.biases = self.biases.copy()
        return d
    
    def save(self, name, i):    
        name, suffix = name.split('.')
        with open(name+"_weights"+str(i)+'.'+suffix, "wb") as file:
            pkl.dump(self.weights, file)        
        with open(name+"_biases"+str(i)+'.'+suffix, "wb") as file:
            pkl.dump(self.biases, file)

    @staticmethod
    def load(name, layer, i):
        name, suffix = name.split('.')
        with open(name+"_weights"+str(i)+'.'+suffix, "rb") as file:
            weights = pkl.load(file)
        with open(name+"_biases"+str(i)+'.'+suffix, "rb") as file:
            biases = pkl.load(file)
        layer.weights = weights
        layer.biases = biases
        return layer

class VarianceScaling:
    def __init__(self, scale, mode, distribution):
        self._validate_inputs(scale, mode, distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

    @staticmethod
    def _validate_inputs(scale, mode, distribution):
        if not isinstance(scale, (int, float)):
            raise Exception(f"Expecting (int, float), got {type(scale)}")
        
        if scale <= 0:
            raise Exception(f"Scale should be greater than 0, got {scale}")
        
        if mode not in ('fan_avg', 'fan_in'):
            raise Exception(f"Mode should be fan_avg or fan_in, got {mode}")
        
        if distribution not in ('uniform', 'normal'):
            raise Exception(f"Distribution should be uniform or normal, got {distribution}")
        
    def get_weights(self, fan_in, fan_out):
        fan = fan_in if self.mode == 'fan_in' else (fan_in + fan_out) / 2
        if self.distribution == 'uniform':
            r = tf.sqrt(tf.constant(3/fan, dtype=tf.float32))
            return tf.random.uniform((fan_in, fan_out), minval=-r, maxval=r)
        
        std = self.scale / fan
        return tf.random.normal((fan_in, fan_out),mean=0, stddev=std)

class BatchNormalization:
    def __init__(self, momentum=0.999, trainable = True):
        self.momentum = momentum
        self.trainable = trainable

    def forward(self, inputs, *args, **kwargs):
        learning = kwargs.get('learning')
        if learning:
            if not hasattr(self, 'size'):
                self.set_size(inputs.shape[1])

            bmean = tf.math.reduce_mean(inputs, axis=0)
            bvar = tf.math.reduce_std(inputs, axis=0) ** 2
            if self.trainable:
                self._update_running_params(bmean, bvar)

            self.std = tf.math.sqrt(bvar + 1e-7)
            self.centered = inputs - bmean
            self.normalized = self.centered / self.std
            return self.gamma * self.normalized + self.beta

        else:
            # TODO see documentation formula?
            return (inputs - self.rmean) / tf.math.sqrt(self.rvar + 1e-7)

    def _update_running_params(self, bmean, bvar):
        self.rmean = self.rmean * self.momentum + bmean * (1 - self.momentum)
        self.rvar = self.rvar * self.momentum + bvar * (1 - self.momentum)

    def backward(self, dvalues, *args, **kwargs):
        m = dvalues.shape[0]

        dbeta = tf.math.reduce_sum(dvalues, axis=0)
        dgamma = tf.math.reduce_sum(dvalues * self.normalized, axis=0)

        self.gamma -= 0.01 * dgamma
        self.beta -= 0.01 * dbeta

        # FIXME 
        # self.optimizer.update([self.beta, self.gamma], [dbeta, dgamma], self.trainable)
        dnorm = dvalues * self.gamma

        dvar = tf.math.reduce_sum(dnorm * self.centered, axis=0) * - 0.5 * self.std**-3
        dmean = tf.math.reduce_sum(dnorm * -self.std**-1, axis=0) + dvar * tf.math.reduce_mean(-2.0 * self.centered, axis=0) 

        dinputs = (dnorm * self.std**-1) + (dvar * 2.0 * self.centered / m) + (dmean / m)
        return dinputs

    def set_size(self, size):
        self.gamma = tf.ones(size)
        self.beta = tf.zeros(size)
        self.rmean = tf.zeros(size)
        self.rvar = tf.zeros(size)
        self.size = size

    def get_size(self):
        return self.size
    
    def copy(self):
        bn = BatchNormalization()
        bn.__dict__ = self.__dict__.copy()
    
def bn_convert(model):
    model = model.copy()
    if not hasattr(model, 'layers'):
        raise Exception("Should have layers")
    
    if not any((isinstance(l, BatchNormalization) for l in model.layers)):
        return model

    dense = None
    idx = set()
    for i, l in enumerate(model.layers):
        if dense is None and isinstance(l, Dense):
            dense = l
            
        elif isinstance(l, BatchNormalization):
            if not dense:
                raise Exception("Current implementation requires BN to proceed Dense")
            idx.add(i)
            std = tf.math.sqrt(l.rvar)
            dense.weights = l.gamma * dense.weights / std
            dense.biases = l.gamma * (dense.biases - l.rmean) / std + l.beta
            dense = None

    model.layers = [l for i, l in enumerate(model.layers) if i not in idx]
    return model

def empty_regularizer(dvalues, w):
    return dvalues

def l1(alpha):
    def backward(dvalues, w):
        dvalues_regularized = dvalues + alpha * tf.math.sign(w)
        return dvalues_regularized
    return backward

def l2(alpha):
    def backward(dvalues, w):
        dvalues_regularized = dvalues + 2 * alpha * w
        return dvalues_regularized
    return backward

def l1_l2(alpha1, alpha2):
    def backward(dvalues, w):
        l1_grad = alpha1 * tf.sign(w)
        l2_grad = 2 * alpha2 * w
        dvalues_regularized = dvalues + l1_grad + l2_grad
        return dvalues_regularized
    return backward

class Dropout:
    def __init__(self, rate):
        if not (0 <= rate <= 1):
            raise Exception("Rate should be between 0 and 1")
        self.rate = rate

    def forward(self, inputs, *args, **kwargs):
        learning = kwargs.get('learning')
        if learning:
            cols = inputs.shape[1]
            n_drop = int(self.rate * cols)
            i = tf.constant(range(cols))
            drop_i = tf.random.shuffle(i)[:n_drop]
            inputs = inputs.copy()
            inputs[:, drop_i] = 0
            inputs /= (1 - self.rate)
            self.prev_dropped = drop_i
        return inputs

    def set_size(self, size):
        self.size = size

    def get_size(self):
        return self.size
    
class AlphaDropout:
    def __init__(self, rate, alpha=-1.7580993408473766, scale=1.0507009873554802):
        if not (0 <= rate <= 1):
            raise Exception("Rate should be between 0 and 1")
        self.rate = rate
        self.alpha = alpha
        self.scale = scale

    def forward(self, inputs, *args, **kwargs):
        learning = kwargs.get('learning')
        if learning:
            cols = inputs.shape[1]
            n_drop = int(self.rate * cols)
            i = tf.constant(range(cols))
            drop_i = tf.random.shuffle(i)[:n_drop]
            inputs = inputs.copy()
            inputs[:, drop_i] = self.alpha
            inputs *= self.scale
            inputs /= (1 - self.rate)
            self.prev_dropped = drop_i
        return inputs

    def set_size(self, size):
        self.size = size

    def get_size(self):
        return self.size
    
def max_norm(norm):
    def backward(weights):
        w_norm = tf.linalg.norm(weights)
        if w_norm > norm:
            weights *= norm / w_norm
    return backward

class Lambda:
    def __init__(self, func):
        self.func = func

    def forward(self, inputs, *args, **kwargs):
        self.inputs = inputs
        return self.func(inputs)

    def set_size(self, size):
        self.size = size

    def get_size(self):
        return self.size

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), kernel_initializer="glorot_uniform", activation=None):
        self.num_filters = filters
        self.kernel_size = kernel_size
        self._validate_kernel_size(kernel_size)
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self._init_activation(activation)

    def _init_activation(self, activation):
        if activation is None:
            self.activation = EmptyActivation() 
        elif isinstance(activation, str):
            act = ACTIVATIONS.get(activation)
            if not act:
                raise Exception(f"Invalid activation {activation}")
            self.activation = act()
        else:
            self.activation = activation

    def build(self):
        h, w = self.size[:2]
        self.output_size = (h - self.kernel_size[0] + 1, w -self.kernel_size[1] + 1, self.num_filters)
        
        if len(self.size) == 2:        
            self.filters = tf.Variable(tf.random.normal((self.num_filters, self.kernel_size[0], self.kernel_size[1]), mean=0, stddev=1/ (self.kernel_size[0] * self.kernel_size[1])))
        else:
            self.filters = tf.Variable(tf.random.normal((self.num_filters, self.kernel_size[0], self.kernel_size[1], self.size[-1]), mean=0, stddev=1/ (self.kernel_size[0] * self.kernel_size[1])))

        self.biases = tf.Variable(tf.zeros(self.num_filters))

    def _validate_kernel_size(self, kernel_size):
        self.kernel_size = kernel_size

    def iterate_regions(self, image, h, w):
        for i in range(h - self.kernel_size[0] + 1):
            for j in range(w - self.kernel_size[1] + 1):
                # im_region = image[i:(i + 3), j:(j + 3)]
                im_region = image[:, i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    @staticmethod
    def _get_image_shape(inputs):
        s = inputs.shape
        if len(s) == 3:
            return s
        if len(s) == 4:
            return s[:3]
        raise Exception(f"Invalid shape {s};")
    
    def call(self, inputs, *args, **kwargs):
        batch_size, h, w = self._get_image_shape(inputs)
        output = tf.zeros((batch_size, h - self.kernel_size[0] + 1, w - self.kernel_size[1] + 1, self.num_filters))
        print(batch_size)

        for im_region, i, j in self.iterate_regions(inputs, h, w):
            f_expanded = tf.expand_dims(self.filters, axis=1) 
            f_tiled = tf.tile(f_expanded, [1, batch_size, 1, 1])
            res = im_region * f_tiled
            res_t = tf.transpose(res, perm=[1, 0, 2, 3])
            final_res = tf.reduce_sum(res_t, axis=[2,3]) + self.biases
            indices = tf.constant([[b, i, j] for b in range(batch_size)])
            output = tf.tensor_scatter_nd_update(output, indices, final_res)

        output = tf.constant(output, dtype=tf.float32)
        return self.activation.forward(output)

    def get_size(self):
        return self.output_size

    def set_size(self, size):
        self.size = size
        self.build()

class MaxPool2D(Layer):
    def __init__(self, pool_size=(2,2)):
        self.pool_size = pool_size

    def iterate_regions(self, image, h, w):
        step_h = self.pool_size[0]
        step_w = self.pool_size[1]
        for i in range(w):
            for j in range(h):
                im_region = image[(i * 2):(i * 2 + step_h), (j * 2):(j * 2 + step_w)]
                yield im_region, i, j

    @staticmethod
    def _get_image_shape(inputs):
        s = inputs.shape
        if len(s) == 3:
            return s
        if len(s) == 4:
            return s[:3]
        raise Exception(f"Invalid shape {s};")
    
    def call(self, inputs, *args, **kwargs):
        batch_size, h, w = self._get_image_shape(inputs)

        new_h = h // 2
        new_w = w // 2

        output = tf.zeros((batch_size, new_h, new_w, self.size[-1]))

        # FIXME vectorize the operation instead of iterating batch size
        for b in range(batch_size):
            image = inputs[b]
            for im_region, i, j in self.iterate_regions(image, new_h, new_w):
                m = tf.reduce_max(im_region, axis=[0, 1])
                output = tf.tensor_scatter_nd_update(output, [[b, i, j]], [m])
        return tf.Variable(output, dtype=tf.float32)

    def get_size(self):
        return self.output_size

    def set_size(self, size):
        self.size = size
        self.build()

    def build(self):
        return