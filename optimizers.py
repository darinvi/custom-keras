from abc import ABC, abstractmethod
from callbacks import ExponentialDecay, OneCycleScheduler
import numpy as np
import tensorflow as tf

class Optimizer(ABC):
    def __init__(self, parent=None, learning_rate=0.01, clipnorm=None, clipvalue=None, weight_decay=None):
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.weight_decay = weight_decay
        self.parent = parent
        if not any((clipnorm, clipvalue)):
            self.clipvalue = 1

    @abstractmethod
    def _update(self, layer, grads, *args, **kwargs):
        pass

    # def update(self, layer, grads, *args, **kwargs):
    #     if not layer.trainable:
    #         return
    #     self._clip_grads(grads)
    #     self._update(layer, grads, *args, **kwargs)

    # FIXME migrate to tf
    def _clip_grads(self, grads):
        for g in grads:
            if self.clipvalue:
                g = np.clip(g, -self.clipvalue, self.clipvalue)

            elif self.clipnorm:
                norm = np.linalg.norm(g)
                if norm > self.clipnorm:
                    g *= self.clipnorm / norm

    def apply_gradients(self, grads):
        # self._clip_grads(grads)
        for g in grads:
            self._update(g)

    # def update(self, grads):
    #     if self.parent is None:
    #         raise Exception("Should set parent model")
        
    #     # self._clip_grads(grads)
    #     dense_layers = [l for l in self.parent.layers if hasattr(l, 'weights')]
    #     for i, l in enumerate(dense_layers):
    #         if not l.trainable:
    #             continue
    #         w_grad = grads[2*i]
    #         b_grad = grads[2*i+1]
    #         self._update(l, [w_grad, b_grad])
    
class SGD(Optimizer):
    def __init__(self, momentum=0, nestrov=False, decay=None, learning_rate=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0 <= momentum <= 1):
            raise Exception("Momentum should be between 0 and 1")
        self.momentum = momentum
        self.velocities = {}
        self.nestrov = nestrov
        self.decay = decay
        self.steps_elapsed = 1
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.step = 1

    def _get_learning_rate(self):
        if isinstance(self.learning_rate, ExponentialDecay):
            return self.learning_rate.get_learning_rate(self.steps_elapsed)
        return self.learning_rate

    def on_batch_end(self, batch, logs=None):
            if self.decay:
                self.learning_rate = self.initial_lr * (1 / (1 + self.decay * self.step))
                self.step += 1

    def _update(self, grads, *args, **kwargs):
        learning_rate = self._get_learning_rate()
        grad, param = grads
        v = self._compute_velocity(learning_rate, param, grad, self.momentum)
        param = self._add(param, self.nestrov, self.momentum, v, learning_rate, grad)
        self.steps_elapsed += 1

    @staticmethod
    def _add(param, nesterov, m, v, lr, g):
        if nesterov:
            res = param.assign_add(param + m * v - lr * g)
        else:
            res = param.assign_add(v)
        return res

    @staticmethod
    def _compute_velocity(lr, p, g, m):
        if not m:
            return -lr * g
        
        # id_ = id(p)
        # v = self.velocities.get(id_)
        # if v is None:
        #     v = tf.Variable(tf.zeros_like(p)) 
        
        # v = self.momentum * v - lr * g
        # self.velocities[id_] = v
        # return v
    
    # def _compute_velocity(self, lr, p, g, m):
    #     if not hasattr(self, 'momentum'):
    #         return -lr * g
        
    #     id_ = id(p)
    #     v = self.velocities.get(id_)
    #     if v is None:
    #         v = tf.Variable(tf.zeros_like(p)) 
        
    #     v = self.momentum * v - lr * g
    #     self.velocities[id_] = v
    #     return v
    
class RMSProp(Optimizer):
    def __init__(self, rho=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = rho
        self.scales = {}
        self.learning_rate = 0.001

    def _update(self, layer, grads, *args, **kwargs):
        s_w = self._compute_scales(layer.weights, grads[0])
        s_b = self._compute_scales(layer.biases, grads[1])
        self._update_params(layer.weights, grads[0], s_w)
        self._update_params(layer.biases, grads[1], s_b)

    def _update_params(self, params, grads, s):
        params -= self.learning_rate * grads / np.sqrt(s + 1e-7)

    def _compute_scales(self, p, g):
        id_ = id(p)
        s = self.scales.get(id_)
        if s is None:
            s = np.zeros_like(p)
        
        r = self.rho
        s = r * s + (1 - r) * g ** 2
        self.scales[id_] = s
        return s
    
class Adam(Optimizer):
    def __init__(self, beta_1=0.9, beta_2=0.999, max=False, weight_decay=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.scales = {}
        self.velocities = {}
        self.max = max
        self.learning_rate = 0.001
        self.weight_decay = weight_decay

    def _update(self, layer, grads, *args, **kwargs):
        s_w = self._compute_scales(layer.weights, grads[0])
        s_b = self._compute_scales(layer.biases, grads[1])
        v_w = self._compute_velocity(layer.weights, grads[0])
        v_b = self._compute_velocity(layer.biases, grads[1])
        ep = args[0]
        self._update_params(layer.weights, s_w, v_w, ep)
        self._update_params(layer.biases, s_b, v_b, ep)

    def _update_params(self, params, s, v, ep):
        v_hat = v / (1 - self.beta_1 ** ep + 1)
        if self.max:
            params += self.learning_rate * v_hat / s
        else:
            s_hat = s / (1 - self.beta_2 ** ep + 1)
            params += self.learning_rate * v_hat / np.sqrt(s_hat + 1e-7)

        if self.weight_decay:
            params *= self.weight_decay

    def _compute_scales(self, p, g):
        id_ = id(p)
        s = self.scales.get(id_)
        if s is None:
            s = np.zeros_like(p)
        
        b2 = self.beta_2
        if self.max:
            s = np.maximum(b2*s, np.abs(g))
        else:
            s = b2 * s + (1 - b2) * g ** 2
        self.scales[id_] = s
        return s
    
    def _compute_velocity(self, p, g):
        id_ = id(p)
        v = self.velocities.get(id_)
        if v is None:
            v = np.zeros_like(p) 
        
        b1 = self.beta_1
        v = b1 * v - (1 - b1) * g
        self.velocities[id_] = v
        return v