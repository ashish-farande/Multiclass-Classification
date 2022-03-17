################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2022
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import math
epsilon = 1e-10
def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    TODO: Normalize your inputs here to have 0 mean and unit variance.
    """
    mean = np.mean(inp, axis = 0)
    std = np.std(inp, axis = 0)
    
    return (inp-mean)/(std**2 + epsilon)**0.5


def one_hot_encoding(labels, num_classes=10):
    """
    TODO: Encode labels using one hot encoding and return them.
    """
    return np.eye(np.max(labels) + 1)[labels]


def load_data(path, mode='train'):
    """
    Load CIFAR-10 data.
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, "cifar-10-batches-py")

    if mode == "train":
        images = []
        labels = []
        for i in range(1,6):
            images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
            data = images_dict[b'data']
            label = images_dict[b'labels']
            labels.extend(label)
            images.extend(data)
        normalized_images = normalize_data(images)
        one_hot_labels    = one_hot_encoding(labels, num_classes=10) #(n,10)
        return np.array(normalized_images), np.array(one_hot_labels)
    elif mode == "test":
        test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
        test_data = test_images_dict[b'data']
        test_labels = test_images_dict[b'labels']
        normalized_images = normalize_data(test_data)
        one_hot_labels    = one_hot_encoding(test_labels, num_classes=10) #(n,10)
        return np.array(normalized_images), np.array(one_hot_labels)
    else:
        raise NotImplementedError(f"Provide a valid mode for load data (train/test)")


def softmax(x):
    """
    TODO: Implement the softmax function here.
    Remember to take care of the overflow condition.
    x: n_sample * n_category
    
    output:
    sm: n_sample * n_category
    """
    
    x -= np.max(x, axis = 1, keepdims = True)
    sm = (np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True))
    
    return sm


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()
        
        elif self.activation_type == "leakyReLU":
            grad = self.grad_leakyReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        return 1/ (1+np.exp(-x))

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        return np.where(x>0,x,0)

    def leakyReLU(self, x, a = 0.1):
        """
        TODO: Implement leaky ReLU here.
        """
        self.leakyrelu_scale = a
        return np.where(x>0,x,a*x)

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        # sigmoid' = sigmoid * (1-sigmoid)
        return self.sigmoid(self.x)*(1-self.sigmoid(self.x))

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        """
        return 1-(np.tanh(self.x))**2

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        """
        return np.where(self.x>0,1,0) 

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        """
        return np.where(self.x>0,1,self.leakyrelu_scale) 

small = 10**(-11)
def cross_entropy(t, y):
    '''
    t: n_sample * n_category
    y: n_sample * n_category
    
    '''
    y = np.clip(y, small, 1-small)
    return (-np.sum(t * np.log(y)))/(t.shape[0])/(t.shape[1])

class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)
        self.b = np.random.randn(1,out_units)  # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        
        self.d_w_m = 0
        self.d_b_m = 0

    def __call__(self, x, targets = None):
        """
        Make layer callable.
        """
        self.targets = targets
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        
        self.a = n_sample * layer_out_dim
        """
        self.x = x
        self.a = x @ self.w + self.b # to next layer
        return self.a # (n_sample * in_dim)(in_dim*out_dim)
        

    def backward(self, delta):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        
        # compute gradient for its own w
        z = self.x # n_sample * in_dim

        self.d_w  = (-z.T @ delta)/self.x.shape[0]  # sum over all samples (in_dim * n_sample)(n_sample * out_dim)
        
        self.d_b = -np.sum(delta, axis = 0)/self.x.shape[0]
        
        # make delta for the incoming layer
        self.d_x = delta @ self.w.T #(n_sample * outdim) * (outdim * indim)
        
        return self.d_x
        
        
        #raise NotImplementedError("Backprop for Layer not implemented.")


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        # the layers does not include softmax
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))
       
        # transform the last layer to softmax?
        self.lr = config['learning_rate']
        self.batch_size = config['batch_size']
        
        
        self.epochs = config['epochs']
        self.early_stop = config['early_stop']
        self.early_stop_epoch = config['early_stop_epoch']
        
        self.L2_penalty = config['L2_penalty']
        self.momentum = config['momentum']
        self.momentum_gamma = config['momentum_gamma']
        # Number of epochs to train the model
                

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        self.targets = targets
        return self.forward(x)

    
    def forward(self, x):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        
        x: np.array n_sample * n_dim
        output(y_hat): softmaxed ans: n_sample * 1
        
        """
        self.x = x

        ans = x
        
        for i, l in enumerate(self.layers): # layer, actication, layer, activation, layer
            if isinstance(l, Layer):
                ans = normalize_data(ans)
            ans = l(ans)
        
        self.y = softmax(ans)
            
        return self.y

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        self.targets = targets
        return cross_entropy(targets, logits)


    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''

        delta = (self.targets - self.y)/self.y.shape[1]
        for i,l in enumerate(self.layers[::-1]): # starting from the last layer
            
            delta = l.backward(delta) # compute delta and gradient for that layer!

        #raise NotImplementedError("Backprop not implemented for NeuralNetwork")
    def update(self):
        
        for i in self.layers:
            if isinstance(i, Layer):
                i.w -= self.lr * (i.d_w + self.L2_penalty * 2 * i.w)
                i.b -= self.lr * (i.d_b + self.L2_penalty * 2 * i.b)
                if self.momentum:
                    old_w = i.d_w_m
                    old_b = i.d_b_m
                    
                    i.d_w_m = self.momentum_gamma * i.d_w_m + self.lr * (i.d_w + self.L2_penalty * 2 * i.w)
                    i.d_b_m = self.momentum_gamma * i.d_b_m + self.lr * (i.d_b + self.L2_penalty * 2 * i.b)

                    i.w -= self.momentum_gamma * old_w
                    i.b -= self.momentum_gamma * old_b


def accuracy(y_hat, y):
    
    return len(np.where(np.argmax(y_hat, axis = 1)==np.argmax(y, axis = 1))[0])/(y.shape[0])

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    train_losses = []
    val_losses = []
    
    train_accs= []
    val_accs = []
    
    # initializing for early stopping
    current_wait = 0
    last_val_loss = math.inf
    early_stopping_happened = False
    
    for i in range(model.epochs):
        
        # minibatch
        index = np.arange(x_train.shape[0])
        np.random.shuffle(index)
        
        yhats_train = []
        ys_train = []
        for n_index in range(0, x_train.shape[0], model.batch_size):
            
            batch_index = index[n_index:n_index+model.batch_size]
            
            x_in = x_train[batch_index,:]
            y_in = y_train[batch_index]
            
            y_hat = model.forward(x_in)
            train_loss = model.loss(y_hat, y_in)
            
            yhats_train.append(y_hat)
            ys_train.append(y_in)
            model.backward()
            model.update()
        
        
        
        
        y_hat = model.forward(x_valid)
        val_loss = model.loss(y_hat, y_valid)
        val_acc = accuracy(y_hat, y_valid)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        train_loss = model.loss(np.concatenate(yhats_train), np.concatenate(ys_train))
        train_acc = accuracy(np.concatenate(yhats_train), np.concatenate(ys_train))
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        
        # check for early stopping
        if last_val_loss < val_loss:
            current_wait += 1
            if current_wait > model.early_stop_epoch and model.early_stop:
                print(f'early_stopped at epoch {i}')
                break
        else:
            current_wait = 0
        
        
        last_val_loss = val_loss

        if i%20 == 0:
            print(f'epoch{i}, val={val_acc}, train_acc={train_acc}')
                    
        
            
    return train_losses, val_losses, train_accs, val_accs
            


def test(model, X_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """

    y_hat = model.forward(X_test)
    
    return accuracy(y_hat, y_test)

    # raise NotImplementedError("Test method not implemented")

def plot(train, validation, title, ylabel=None):
    plot.fig_count += 1
    plt.plot(train, color='y', label='Train')
    plt.plot(validation, color='g', label='Validation')
    plt.title(title)
    plt.legend()
    plt.xlabel('Epochs')
    if ylabel is None:
        ylabel = title
    plt.ylabel(ylabel)
    plt.savefig('Fig_' + str(plot.fig_count) + '.png')
    plt.show()

plot.fig_count = 0
if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_input, y_input = load_data(path="./data/", mode="train")
    x_test,  y_test  = load_data(path="./data/", mode="test")
    
    # TODO: Create splits for validation data here.
    # x_val, y_val = ...
    ind  = int(x_input.shape[0]*0.99)
    x_train, x_valid = x_input[:ind,:], x_input[ind:,:]
    y_train, y_valid = y_input[:ind,:], y_input[ind:,:]

    # TODO: train the model
    train_losses, val_losses, train_accs, val_accs = train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)

    # TODO: Plots
    print("Accuracy: ", test_acc)
    plot(train_losses, val_losses, 'Loss')
    plot(train_accs, val_accs, 'Loss')

    
