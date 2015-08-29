'''
A simple Neural net wrapper, based on nolearn >> lasagne >> theano
'''

import numpy as np

from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import tanh
from nolearn.lasagne import NeuralNet

# a helper class to cache the best model and stop early if the
# model isn't improving anymore on the validation set
class EarlyStopping(object):
    def __init__(self, patience):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print "Early stopping."
            print "Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch)

            nn.load_params_from(self.best_weights)
            raise StopIteration()

# another helper class to adjust the learning rate and the momentum
class AdjustVariable(object):
    def __init__(self, name, stop, decrement=0.0001, increment=None):
        self.name = name
        self.stop = stop
        self.decrement = decrement
        self.increment = increment

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch']
        if self.increment:
            new_value = min(getattr(nn, self.name) + self.increment, self.stop)
        else:
            new_value = max(getattr(nn, self.name) - self.decrement, self.stop)
        nn.__dict__[self.name] = np.cast['float32'](new_value)

class NN(object):
    
    def __init__(self, input_size, hidden_1_size, hidden_2_size=None):
        n_layers = [
            ('input', layers.InputLayer),
            ('hidden1', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer)
        ]
        if hidden_2_size is not None:
            n_layers.extend(
                [('hidden2', layers.DenseLayer), ('dropout2', layers.DropoutLayer)]
            )
        n_layers.append(('output', layers.DenseLayer))
        
        self.model = NeuralNet(
            layers=n_layers,
            input_shape=(None, input_size),
            hidden1_num_units=hidden_1_size, dropout1_p=0.5,
    
            output_nonlinearity=tanh,
            output_num_units=1,
            regression=True,

            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,
    
            eval_size=0.1,
            on_epoch_finished=[
                AdjustVariable('update_learning_rate', stop=0.0001, decrement=0.00001),
                AdjustVariable('update_momentum',      stop=0.999,  increment=0.0001),
                EarlyStopping(patience=100)
            ],
            
            max_epochs=5000,
            verbose=1
        )
        if hidden_2_size is not None:
            self.model.__dict__['hidden2_num_units'] = hidden_2_size
            self.model.__dict__['dropout2_p'] = 0.5            
    
    def train(self, X, Y):
        self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32))
    
    def predict_continuous(self, X_test):
        return self.model.predict(np.asarray(X_test, dtype=np.float32))
    
    def predict_classes(self, X_test):
        Y_pred = self.predict_continuous(X_test)
        
        # threshold the continuous values to get the classes
        pos = Y_pred >= .33
        neg = Y_pred <= -0.33
        neu = np.logical_and(Y_pred < 0.33, Y_pred > -0.33)
        Y_pred[pos] = 1
        Y_pred[neg] = -1
        Y_pred[neu] = 0
        
        return Y_pred.reshape(-1)