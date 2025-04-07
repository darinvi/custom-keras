import sys
import pickle
import numpy as np

class Callback:
    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

class History(Callback):
    def __init__(self):
        self.logs = []

    def on_epoch_end(self, epoch, logs):
        losses = logs.get('losses')
        metrics = logs.get('metrics')
        show_log = logs.get('show_log')
        time = logs.get('time')
        return self._msg(losses, metrics, show_log, epoch, time)

    # FIXME validation metric 2 times.?
    def _msg(self, losses, metrics, show_log, epoch, time):
        msg = f'epoch {epoch + 1}/{self.n_epochs} - time: {time:.2f}s - '
        msg += self._metrics_msg(metrics)
        msg += f' - loss: {losses[0]:.5f}'
        if len(losses) == 2:
            msg += f' - loss_validation: {losses[1]:.5f}'
        
        if show_log:
            sys.stdout.write(msg +'\n')
        
    @staticmethod
    def _metrics_msg(metrics, val=False):
        msg = []
        for m in metrics:
            metric, value = m.name, m.result()
            msg.append(f'{metric}: {value:.5f}')

        return ' '.join(msg)
    
def validate_loss(logs):
    losses = logs.get('losses')
    if not losses:
        raise Exception("Should have losses")
    return losses[-1] #len(losses) is either 1 or 2, if two -> interested in val. loss

def validate_parent_layers(logs):
    parent = logs.get('parent')
    if not parent:
        raise Exception("Invalid log")
    
    if not hasattr(parent, 'layers'):
        raise Exception("Parent should have layers")
    
    return parent, parent.layers

class ModelCheckpoint(Callback):
    def __init__(self, dir, save_best_only=False):
        self.dir = dir
        self.save_best_only = save_best_only
        self.best_loss = None

    def on_epoch_end(self, epoch, logs=None):        
        loss = validate_loss(logs)
        if self.best_loss is None:
            self.best_loss = loss

        if loss > self.best_loss:
            return
         
        self.best_loss = loss
        _, layers = validate_parent_layers(logs)
        self._save_weights(epoch, layers)

    def _save_weights(self, epoch, layers):
        if not layers:
            raise Exception("No layers passed to ModelCheckpoint")
        
        if not any((hasattr(l, 'weights') for l in layers)):
            raise Exception("No learning layers passed to ModelCheckpoint")
        
        counter = 0
        for l in layers:
            if not hasattr(l, 'weights'):
                continue

            name_w = f'{self.dir}/weights_layer_{counter}'
            name_b = f'{self.dir}/biases_layer_{counter}'

            if not self.save_best_only:
                name_w += f'_epoch_{epoch}'
                name_b += f'_epoch_{epoch}'

            with open(f'{name_w}.pkl', 'wb') as f:
                pickle.dump(l.weights, f)

            with open(f'{name_b}.pkl', 'wb') as f:
                pickle.dump(l.biases, f)

            counter += 1

class EarlyStopping(Callback):
    def __init__(self, patience = 10, restore_best_weights=False):
        if patience <= 0:
            raise Exception(f"Patience should be above 0, not {patience}")
        
        self.patience = patience
        self.counter = 0
        self.restore_best_weights = restore_best_weights
        self.best_loss = None

    def _track_best_weights(self, layers):
        self.best_weights = []
        self.best_biases = []
        for l in layers:
            if not hasattr(l, 'weights'):
                continue
            self.best_weights.append(l.weights.copy())
            self.best_biases.append(l.biases.copy())

    def _restore_best_weights(self, parent):
        for l in reversed(parent.layers):
            if not hasattr(l, 'weights'):
                continue
            l.weights = self.best_weights.pop()
            l.biases = self.best_biases.pop()

    def on_epoch_end(self, epoch, logs=None):
        loss = validate_loss(logs)

        if self.best_loss is None:
            self.best_loss = loss

        if loss > self.best_loss:
            self.counter += 1
        else:
            self.best_loss = loss
            self.counter = 0

        parent, layers = validate_parent_layers(logs)
        if self.counter == self.patience:
            parent.stop_training = True
            if not hasattr(parent, 'stop_training'):
                raise Exception("EarlyStopping needs stop_training attr")
            
            if self.restore_best_weights:
                self._restore_best_weights(parent)

        if self.restore_best_weights:
            self._track_best_weights(layers)

class LearningRateScheduler(Callback):
    def __init__(self, func):
        if not hasattr(func, '__call__'):
            raise Exception("Should be callable")
        
        self.func = func

    def on_epoch_end(self, epoch, logs=None):
        parent = logs.get("parent")
        if not parent:
            raise Exception("No parent")
        
        optimizer = parent.optimizer
        lr = optimizer.initial_lr
        optimizer.learning_rate = self.func(epoch, lr)

class ReduceLROnPlateau(Callback):
    def __init__(self, factor=0.1, patience=10):
        self.factor = factor
        self.patience = patience
        self.best_loss = None

    def on_epoch_end(self, epoch, logs=None):
        loss = validate_loss(logs)

        if self.best_loss is None:
            self.best_loss = loss

        if loss <= self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return
        
        self.counter += 1
        
        if self.counter != self.patience:
            return
        
        parent = logs.get("parent")
        if not parent:
            raise Exception("No parent")
        parent.optimizer.learning_rate *= self.factor


class ExponentialDecay:
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def get_learning_rate(self, step):
        return self.initial_learning_rate * (self.decay_rate ** (step / self.decay_steps))
    

class OneCycleScheduler(Callback):
    def __init__(self, initial_learning_rate, max_lr, steps, magnitute=0.1):
        self.initial_learning_rate = initial_learning_rate
        self.max_lr = max_lr
        self.midway = steps // 2
        self.update = (max_lr - initial_learning_rate) / self.midway
        self.step = 0
        self.magnitude = magnitute

    def on_batch_end(self, batch, logs=None):
        parent = logs.get("parent")
        if not parent:
            raise Exception("No parent")

        max_epochs = logs.get('max_epochs')
        if not max_epochs:
            raise Exception("No max epochs")
        
        epoch = logs.get('epoch')
        if not max_epochs:
            raise Exception("No epoch")

        last_epochs = int(0.1 * max_epochs)
        if max_epochs - epoch <= last_epochs:
            parent.optimizer.learning_rate *= self.magnitude
            return
        
        if self.step <= self.midway:
            parent.optimizer.learning_rate += self.update
        else:
            parent.optimizer.learning_rate -= self.update
        
        self.step +=1

class DetailedStats(Callback):
    def on_epoch_end(self, epoch, logs=None):
        layers = logs.get("layers")
        if not layers:
            raise Exception("No layers provided")
        i = 1
        for l in layers:
            if not hasattr(l, 'weights'):
                continue
            print(f"epoch {epoch+1} dense_{i}: mean: {np.mean(l.weights):.3f}; std: {np.std(l.weights):.3f}")
            i += 1