from layers import *
from optimizers import *
from metrics import *
from callbacks import History
from time import time

OPTIMIZERS = {
    'sgd': SGD,
}

METRICS = {
    'rmse': RootMeanSquaredError,
    'cross_e': sparse_categorical_crossentropy,
    'accuracy': SparseCategoricalAccuracy,
}

LOSSES = {
    'mse': MSELoss,
    'cat-cross-ent': CategoricalCrossEntropyLoss,
}

class Model:
    def __init__(self, batch_size=32, tol=1e-4):
        self.batch_size = batch_size
        self.history = History()
        self.tol = tol
        self.stop_training = False
        self.extra_losses = []
        self.extra_metrics = []

    @staticmethod
    def _validate_callbacks(callbacks):
        if not callbacks:
            return
        cb_methods = ('on_batch_begin', 'on_batch_end', 'on_epoch_being', 'on_epoch_end')
        for cb in callbacks:
            if not any((x in dir(cb) for x in cb_methods)):
                raise Exception("Invalid callback implementation")                

    def _validation_metrics(self):
        new_metrics = []
        for m in self.metrics:
            new = METRICS.get(m.name)()
            new.name += '_validation'
            new_metrics.append(new)
        self.metrics.extend(new_metrics)

    @staticmethod
    @tf.function(reduce_retracing=True)
    def _fit(model, X_b, y_b):
        with tf.GradientTape() as tape:
            pred = model.forward(X_b, learning=True)
            batch_loss = model.loss.forward(y_b, pred)
            batch_loss += model.get_losses()

        trainable_params = model.trainable_variables()
        gradients = tape.gradient(batch_loss, trainable_params)
        model.optimizer.apply_gradients(zip(gradients, trainable_params))
        return pred

    def fit(self, X, y, epochs=1, validation_data=None, show_log=True, callbacks=[]):
        if validation_data:
            self._validation_metrics()

        self.history.n_epochs = epochs
        self._validate_callbacks(callbacks)
        callbacks.append(self.history)
        prev_loss = None
        for ep in range(epochs):
            st = time()
            self._on_epoch_begin(ep, callbacks) 
            for i in range(0, X.shape[0], self.batch_size):
                self._on_batch_begin(i, callbacks)
                X_b = X[i: i+self.batch_size]
                y_b = y[i: i+self.batch_size]
                pred = self._fit(self, X_b, y_b)
                for metric in self.metrics:
                    if 'validation' not in metric.name:
                        metric(y_b, pred)

                self._on_batch_end(i, callbacks, max_epochs=epochs, epoch=ep)

            loss = self.loss.forward(y_b, pred)
            et = time()
            self._on_epoch_end(validation_data, show_log, loss, callbacks, ep, et - st)
            self.has_converged(prev_loss, loss)
            if self.stop_training:
                break
            prev_loss = loss

            for metric in self.metrics:
                metric.reset_states()

        return self.history

    def trainable_variables(self):
        params = []
        for l in self.layers:
            if isinstance(l, Dense):
                params.append(l.weights)
                params.append(l.biases)
            elif isinstance(l, Conv2D):
                params.append(l.filters)
                params.append(l.biases)
        return params

    def forward(self, X, learning=True):
        inputs = X
        for layer in self.layers:
            inputs = layer.forward(inputs, learning=learning)
        return inputs

    def has_converged(self, prev_loss, loss):
        if not prev_loss:
            self.stop_training = False
            return
        if self.stop_training == True:
            return
        # self.stop_training = abs((prev_loss - loss) / prev_loss) <= self.tol #FIXME check if original like this

    @staticmethod
    def _on_batch_begin(batch, callbacks):
        logs = {}
        for cb in callbacks:
            cb.on_batch_begin(batch, logs)

    def _on_batch_end(self, batch, callbacks, max_epochs, epoch):
        logs = {'parent':self, 'max_epochs': max_epochs, 'epoch':epoch}
        for cb in callbacks:
            cb.on_batch_end(batch, logs)
        
        if hasattr(self.optimizer, 'on_batch_end'):
            self.optimizer.on_batch_end(batch, logs)

    @staticmethod
    def _on_epoch_begin(epoch, callbacks):
        logs = {}
        for cb in callbacks:
            cb.on_epoch_begin(epoch, logs)

    def _on_epoch_end(self, validation_data, show_log, loss, callbacks, epoch, epoch_time):
        losses = [loss]
        if validation_data:
            X_v, y_v = validation_data
            pred_val = self.predict(X_v, learning=True)
            loss_val = self.loss.forward(y_v, pred_val)
            losses.append(loss_val)
            for m in self.metrics:
                if 'validation' in m.name:
                    m(y_v, pred_val)

        logs = {
            'losses': losses, 
            'metrics': self.metrics, 
            'show_log': show_log, 
            'layers': self.layers,
            'parent': self,
            'time': epoch_time
        }

        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)

    def predict(self, X, learning=False):
        return self.forward(X, learning=learning)
    
    def _compute_metrics(self, y_true, y_pred, val = False):
        metrics = []
        for name, m in self.metrics:
            metric = m(y_true, y_pred)
            if val:
                name += '_validation'
            metrics.append((name, metric))
        return metrics

    def compile(self, loss, optimizer, metrics=[]):
        loss_ = LOSSES.get(loss)
        self.loss = loss_()
        if loss_ is None:
            raise Exception(f"Invalid loss function {loss}")
        
        met_ = [METRICS.get(m)() for m in metrics]
        self.metrics = met_
        if not all(met_):
            raise Exception(f"Invalid input form metrics {metrics}")

        if issubclass(optimizer.__class__, Optimizer):
            optimizer.parent = self
            self.optimizer = optimizer
        else:
            opt_ = OPTIMIZERS.get(optimizer)
            if opt_ is None:
                raise Exception(f"Invalid optimizer {loss}")
            self.optimizer = opt_()

        for l in self.layers:
            l.optimizer = self.optimizer

    def get_losses(self):
        loss = tf.reduce_sum(self.extra_losses)
        self.extra_losses = []
        return loss

    def add_loss(self, loss):
        self.extra_losses.append(loss)

    def add_metric(self, name, metric):
        self.extra_metrics.append((name, metric))
        
    def get_metrics(self, metrics):
        metrics.extend(self.extra_metrics)
        self.extra_metrics = []

    def copy(self):
        m = Model(n_epochs=self.n_epochs, batch_size=self.batch_size, tol=self.tol)
        m.__dict__ = self.__dict__.copy()
        layers = []
        for l in self.layers:
            layers.append(l.copy())
        return m
    
    def save(self, name):
        with open(name, "wb") as file:
            pkl.dump(self, file)
        for i, l in enumerate(self.layers):
            l.save(name, i)

    @staticmethod
    def load_model(name):
        with open(name, "rb") as file:
            model = pkl.load(file)
        for i, l in enumerate(model.layers):
            model.layers[i] = l.load(name, l, i)
        return model

class Sequential(Model):
    def __init__(self, layers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.names = []
        self.names_counter = {} #FIXME use Counter
        self.layers = []
        self._init_layers(layers)

    def _init_layers(self, layers):
        if layers is None:
            return
        
        if not isinstance(layers, list):
            raise Exception("should be list of objects/tuples")
        
        if isinstance(layers[0], tuple):
            self.layers = [l[1] for l in layers]
            self.names = [n[0] for n in layers]
        else:
            for l in layers:
                self._add_layer(l)

    def _add_layer(self, l):
        name = l.name if hasattr(l, 'name') else l.__class__.__name__.lower()
        if name not in self.names_counter:
            self.names_counter[name] = 0
        else:
            self.names_counter[name] += 1
            name += f"_{self.names_counter[name]}"
        
        if self.layers:
            l.set_size(self.layers[-1].get_size())

        self.layers.append(l)
        self.names.append(name)

    def get_layer(self, name_):
        if name_ not in self.names_counter:
            raise Exception(f"No layer {name_}")

        for name, l in self.layers:
            if name == name_:
                return l

    def add(self, layer):
        self._add_layer(layer)

    def summary(self):
        res = [['layer(type)', 'Output Shape', 'Param #']]
        prev_shape = 0
        for name, layer in self.layers:
            if isinstance(layer, Flatten):
                shape = reduce(lambda prev, curr: prev * curr, layer.input_shape, 1)
            elif isinstance(layer, Dense):
                shape = layer.n_neurons
            l_summary = [f'{name} ({layer.__class__.__name__})', f'{None}, {shape}', str(prev_shape * shape)]
            res.append(l_summary)
            prev_shape = shape
        return self._format_summary(res)

    @staticmethod
    def _format_summary(res):
        lens = tf.constant([[len(r_) for r_ in r] for r in res])
        max_lens = tf.reduce_max(lens, axis=0)
        margin = 5
        for i in range(3):
            len_ = max_lens[i] + margin
            for j, e in enumerate(res):
                res[j][i] = e[i].ljust(len_)
        return '\n'.join([''.join(r) for r in  res])
    
class MCModel(Sequential):
    def predict(self, X):
        pred = self.predict_proba(X)
        return tf.argmax(pred, axis=1)

    def predict_proba(self, X):
        y_probas = tf.stack([super().predict(X, learning=True) for _ in range(10)])
        y_proba = tf.math.reduce_mean(y_probas, axis=0)
        return y_proba