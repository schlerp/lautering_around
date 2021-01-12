Title: Neural Networks in Numpy
SubTitle: 
Date: 2016-07-13 06:45
Modified: 2020-07-25 15:30
Category: Data Science
Tags: Neural Network, Numpy, Python
Slug: neural-network-numpy
Authors: Schlerp


A while ago a friend of mine mentioned something called an artificial neural network and I remember thinking to myself, "Fuck... they must be difficult to implement". How wrong was I! Let's make some Artificial Neural Networks in python! we will start with some basic implementations in Numpy in this post. After we will move to NumExpr and then once we have the basics i will show you the beautiful Keras library.

So first we need a copy of python, and the Numpy Package installed. Numpy is a science computing library for python. basically it gives you some nice algebraic ways to do maths with python, namely the Matrices (ndarray's) and simple functions for dot products etc. Install numpy with ```pip install numpy```.

Ok! Let's get started! Take a look at this [post](https://iamtrask.github.io/2015/07/12/basic-python-network/) by iamtrask. this is basically their implementation however i have added the ability to add layers in an easier fashion.

First let's import numpy!
```python
import numpy as np
```

Now let's implement a Nonlinear class
```python
class Nonlinear(object):
    """
    Nonlinear
    ---------
    this is used to set up a non linear for a
    network. The idea is you can instantiate it
    and set what type of non linear function it
    will be for that particular neaural network
    """
    
    _FUNC_TYPES = ('sigmoid',
                   'softmax',
                   'relu',
                   'tanh',
                   'softplus')
    
    def __init__(self, func_type='sigmoid'):
        if func_type in self._FUNC_TYPES:
            if func_type == self._FUNC_TYPES[0]:
                # sigmoid
                self._FUNCTION = self._FUNC_TYPES[0]
            elif func_type == self._FUNC_TYPES[1]:
                # softmax
                self._FUNCTION = self._FUNC_TYPES[1]
            elif func_type == self._FUNC_TYPES[2]:
                # relu
                self._FUNCTION = self._FUNC_TYPES[2]
            elif func_type == self._FUNC_TYPES[3]:
                # tanh
                self._FUNCTION = self._FUNC_TYPES[3]
            elif func_type == self._FUNC_TYPES[4]:
                # tanh
                self._FUNCTION = self._FUNC_TYPES[4]
        else:
            # default to sigmoid on invalid choice?
            print("incorrect option `{}`".format(func_type))
            print("defaulting to sigmoid")
            self._init_sigmoid()
    
    def __call__(self, x, derivative=False):
        ret = None
        if self._FUNCTION == self._FUNC_TYPES[0]:
            # sigmoid
            if derivative:
                ret = x*(1-x)
            else:
                try:
                    ret = 1/(1+np.exp(-x))
                except:
                    ret = 0.0
        elif self._FUNCTION == self._FUNC_TYPES[1]:
            # softmax
            if derivative:
                # from below + http://www.derivative-calculator.net/
                ret = 2*(np.exp(x) / np.sum(np.exp(x)))
            else:
                # from: https://gist.github.com/stober/1946926
                #e_x = np.exp(x - np.max(x))
                #ret = e_x / e_x.sum()
                # from: http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
                ret = np.exp(x) / np.sum(np.exp(x))
        elif self._FUNCTION == self._FUNC_TYPES[2]:
            # relu
            if derivative:
                # from below + http://www.derivative-calculator.net/
                ret = 2*(abs(x))
            else:
                ret = x*(abs(x))
        elif self._FUNCTION == self._FUNC_TYPES[3]:
            # tanh
            if derivative:
                # from my own memory of calculus :P
                ret = 1.0-x**2
            else:
                ret = np.tanh(x)
        elif self._FUNCTION == self._FUNC_TYPES[3]:
            # softmax
            if derivative:
                # from wikipedia
                ret = 1.0/(1+np.exp(-x))
            else:
                ret = np.log(1+np.exp(x))
        return ret

```
This class will make it easy to implement a variety of different activation functions!

Next, lets implement iamtrask's example as a class!
```python
class NeuralNetwork(object):
    """
    Neural network
    --------------
    This is my neural netowrk class, it basically holds all 
    my variables and uses my other functions/classes
    """
    def __init__(self, input, hidden, output, non_lin=Nonlinear(), bias=False, alpha=1, ):
        if bias:
            self._BIAS = True
            self._INPUT = input + 1
        else:
            self._BIAS = False
            self._INPUT = input
        self._ALPHA = alpha
        self._HIDDEN = hidden
        self._OUTPUT = output
        self.non_lin = non_lin
        self._init_nodes()

    def _init_nodes(self):
        # set up weights (synapses)
        self.w_in = np.random.randn(self._INPUT, self._HIDDEN) 
        self.w_out = np.random.randn(self._HIDDEN, self._OUTPUT)
        # set up changes
        #self.change_in = np.zeros((self._INPUT, self._HIDDEN))
        #self.change_out = np.zeros((self._HIDDEN, self._OUTPUT))        
        
    def _do_layer(self, layer_in, weights):
        """Does the actual calcs between layers :)"""
        ret = self.non_lin(np.dot(layer_in, weights))
        return ret
    
    #def _error_delta(self, layer_in, y):
        #layer_error = y - layer_in
        #layer_delta = layer_error * self.non_lin(derivative=True)
        #return layer_error, layer_delta
        
    def train(self, x, y, train_loops=1000):
        for i in range(train_loops):

            # from: https://iamtrask.github.io/2015/07/28/dropout/
            
            # Why Dropout: Dropout helps prevent weights from converging to 
            # identical positions. It does this by randomly turning nodes off 
            # when forward propagating. It then back-propagates with all the 
            # nodes turned on.
            # A good initial configuration for this for hidden layers is 50%. 
            # If applying dropout to an input layer, it's best to not exceed 
            # 25%.
            # use Dropout during training. Do not use it at runtime or on your 
            # testing dataset.
            
            #if do_dropout:
                #layer_1 *= np.random.binomial([np.ones((len(X),hidden_dim))],
                                          #1-dropout_percent)[0] * \
                                          #(1.0/(1-dropout_percent))
            
            # set up layers
            layer0 = x
            layer1 = self._do_layer(layer0, 
                                    self.w_in)
            layer2 = self._do_layer(layer1,
                                    self.w_out)
            
            # calculate errors
            layer2_error = y - layer2
            layer2_delta = layer2_error * self.non_lin(layer2, derivative=True)
            
            layer1_error = layer2_delta.dot(self.w_out.T)
            layer1_delta = layer1_error * self.non_lin(layer1, derivative=True)
            
            if (i % (train_loops/10)) == 0:
                print("loop: {}".format(i))
                print("Layer1 Error: {}".format(np.mean(np.abs(layer1_error))))                
                print("Layer2 Error: {}".format(np.mean(np.abs(layer2_error))))
                print("Guess: ")
                print(layer2[0])               
                print("Guess (round): ")
                print(np.round(layer2[0], 1))
                print("Actual: ")
                print(y[0])
                
            #if (i % (train_loops/100)) == 0:
                #print("currently on loop: {}".format(i))
            # backpropagate error
            self.w_out += self._ALPHA * layer1.T.dot(layer2_delta)
            self.w_in += self._ALPHA * layer0.T.dot(layer1_delta)
            
    
    def guess(self, x):
        _in = x
        _hidden = self.non_lin(np.dot(_in, self.w_in))
        _out = self.non_lin(np.dot(_hidden, self.w_out))
        return _out

```
See how its basically the exact same code but wrapped up in a class? This makes it easy to import and pass around inside of an application!

Now, lets look at implementing a multi layer version of this network!
```python
class NNN(object):
    """N-layered neural network"""
    def __init__(self, inputs, weights, outputs, alpha):
        self.inputs = inputs
        self.outputs = outputs
        self._ALPHA = alpha
        self._num_of_weights = len(weights)
        self._LAYER_DEFS = {}
        self.WEIGHT_DATA = {}
        self.LAYER_FUNC = {}
        self.LAYERS = {}
        for i in range(self._num_of_weights):
            #(in, out, nonlin)
            self._LAYER_DEFS[i] = {'in': weights[i][0],
                              'out': weights[i][1],
                              'nonlin': weights[i][2]}
        print(self._LAYER_DEFS)
        self._init_layers()
    
    def _init_layers(self):
        for i in range(self._num_of_weights):
            _in = self._LAYER_DEFS[i]['in']
            _out = self._LAYER_DEFS[i]['out']
            _nonlin = self._LAYER_DEFS[i]['nonlin']
            self.WEIGHT_DATA[i] = np.random.randn(_in, _out)
            self.LAYER_FUNC[i] = _nonlin
    
    def _do_layer(self, prev_layer, next_layer, nonlin):
        """Does the actual calcs between layers :)"""
        ret = nonlin(np.dot(prev_layer, next_layer))
        return ret

    def train(self, x, y, train_loops=100):
        for j in range(train_loops):
            # set up layers
            prev_layer = x
            prev_y = y
            next_weight = None
            l = 0
            self.LAYERS[l] = x
            for i in range(self._num_of_weights):
                l += 1
                next_weight = self.WEIGHT_DATA[i]
                nonlin = self.LAYER_FUNC[i]
                current_layer = self._do_layer(prev_layer, next_weight, nonlin)
                self.LAYERS[l] = current_layer
                prev_layer = current_layer
            last_layer = current_layer
            #print(last_layer)
            #
            #layer2_error = y - layer2
            #layer2_delta = layer2_error * self.non_lin(layer2, derivative=True)
            
            #layer1_error = layer2_delta.dot(self.w_out.T)
            #layer1_delta = layer1_error * self.non_lin(layer1, derivative=True)
            
            #self.w_out += self._ALPHA * layer1.T.dot(layer2_delta)
            #self.w_in += self._ALPHA * layer0.T.dot(layer1_delta)              
            
            # calculate errors
            output_error = y - last_layer
            output_nonlin = self.LAYER_FUNC[self._num_of_weights - 1]
            output_delta = output_error * output_nonlin(last_layer, derivative=True)

            prev_delta = output_delta
            prev_layer = last_layer
            for i in reversed(range(self._num_of_weights)):
                weight = self.WEIGHT_DATA[i]
                current_weight_error = prev_delta.dot(weight.T)
                current_weight_nonlin = self.LAYER_FUNC[i]
                current_weight_delta = current_weight_error * current_weight_nonlin(self.LAYERS[i], derivative=True)
                # backpropagate error
                self.WEIGHT_DATA[i] += self._ALPHA * self.LAYERS[i].T.dot(prev_delta)
                prev_delta = current_weight_delta
                

            if (j % (train_loops/10)) == 0:
                print("loop: {}".format(j))
                #print("Layer1 Error: {}".format(np.mean(np.abs(layer1_error))))                
                #print("Layer2 Error: {}".format(np.mean(np.abs(layer2_error))))
                #print("Guess: ")
                #print(last_layer[0])
                #print("output delta: ")
                #print(np.round(output_delta, 2))
                print("Guess (rounded): ")
                print(np.round(last_layer[0], 1))
                print("Actual: ")
                print(y[0])
        
    def guess(self, x):
        prev_layer = x
        prev_y = y
        next_weight = None
        l = 0
        self.LAYERS[l] = x
        for i in range(self._num_of_weights):
            l += 1
            next_weight = self.WEIGHT_DATA[i]
            nonlin = self.LAYER_FUNC[i]
            current_layer = self._do_layer(prev_layer, next_weight, nonlin)
            self.LAYERS[l] = current_layer
            prev_layer = current_layer
        last_layer = current_layer
        return last_layer
```
see how the training function and the guess function loop through the layers? They pass them to an internal helper function ```_do_layer```.

Now, to make it easy to load the mnist! we will be using an external script from [here](https://gist.github.com/akesling/5358964).

```python
import os
import struct
import numpy as np

# from: https://gist.github.com/akesling/5358964

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def read(dataset="training", path="./data"):
    """
    Python function for importing the MNIST data set. It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise(ValueError, "dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
    
    
def get_flat_mnist(dataset="training", path="./mnist", items=60000, normalize=False):
    images = tuple()
    labels = tuple()
    i = 0
    for image in read(dataset, path):
        images += (image[1],)
        labels += (image[0],)
        i += 1
        if i == items:
            break

    flat_images = tuple()
    for image in images:
        flat_image = []
        for row in image:
            l_row = list(row)
            for item in l_row:
                if normalize:
                    if item <= 127:
                        flat_image.append(0)
                    else:
                        flat_image.append(1)
                else:
                    flat_image.append(item)
        flat_images += (flat_image,)
    del images


    out_labels = tuple()
    # [0,1,2,3,4,5,6,7,8,9]
    for item in labels:
        if item == 0:
            out_labels += ([1,0,0,0,0,0,0,0,0,0],)
        elif item == 1:
            out_labels += ([0,1,0,0,0,0,0,0,0,0],)
        elif item == 2:
            out_labels += ([0,0,1,0,0,0,0,0,0,0],)
        elif item == 3:
            out_labels += ([0,0,0,1,0,0,0,0,0,0],)
        elif item == 4:
            out_labels += ([0,0,0,0,1,0,0,0,0,0],)
        elif item == 5:
            out_labels += ([0,0,0,0,0,1,0,0,0,0],)
        elif item == 6:
            out_labels += ([0,0,0,0,0,0,1,0,0,0],)
        elif item == 7:
            out_labels += ([0,0,0,0,0,0,0,1,0,0],)
        elif item == 8:
            out_labels += ([0,0,0,0,0,0,0,0,1,0],)
        elif item == 9:
            out_labels += ([0,0,0,0,0,0,0,0,0,1],)
    return flat_images, out_labels
```
I've added a function ```get_flat_mnist``` to get a flattened version of the mnist dataset. This basically means that each image of the data has been appended onto the end of the first row in order to produce and 784 item long single row array. The answer row has been adjusted to fit the format that the output neurons will fire. You can see that the positions in the array represent the numbers 0 to 9. The position that has a 1 at it represents the number the image corresponded to!

Now lets quickly run another script from the same folder we are working in. This will load some different sized data sets and pickle them with cpickle (this requires cpickle, or atleast pickle installed, you can do this with ```# pip install cpickle```). 

```python
import mnist
try:
    import cPickle as pickle
except:
    import pickle

def make_mnist_np():
    l_items = [100, 500, 1000, 2000, 5000, 10000, 20000]
    for items in l_items:
        print("grabbing {} data...".format(items))
        t_in, t_out = mnist.get_flat_mnist(items=items, normalize=True)
        print("  got nmist array!")
        print('  {}x{}'.format(len(t_in), len(t_in[0])))
        x = np.array(t_in, dtype=np.float)
        y = np.array(t_out, dtype=np.float)
        with open('mnist/tx{}'.format(items), 'wb+') as f:
            pickle.dump(x, f)
        with open('mnist/ty{}'.format(items), 'wb+') as f:
            pickle.dump(y, f)

if __name__ == '__main__':
    make_mnist_np()
```

This will make sense when we implement the mini menu system in a second. Back in the original file we defined the neural network and the nonlinear function, lets start to use some of the things we have implemented.
```python
if __name__ == '__main__':
    import mnist
    try:
        import cPickle as pickle
    except:
        import pickle


    # get data
    if input("load mnist training data?").lower() == 'y':
        load_d = input("  enter filename (eg. 500 = tx-500, ty-500): ")
        with open("mnist/tx{}".format(load_d), 'rb') as f:
            x = pickle.load(f)
        with open("mnist/ty{}".format(load_d), 'rb') as f:
            y = pickle.load(f)
    else:
        print("grabbing data...")
        t_in, t_out = mnist.get_flat_mnist(items=1000, normalize=True)
        print("  got nmist array!")
        print('  {}x{}'.format(len(t_in), len(t_in[0])))
        x = np.array(t_in, dtype=np.float)
        y = np.array(t_out, dtype=np.float)        
    
    
    load = input("load network? (y/N): ")
    if load.lower() == 'y':
        fname = input("network filename: ")
        with open(fname, 'rb') as f:
            nnn = pickle.load(f)
    else:
        # set hypervariables
        i_input = 784 # this is how many pixel per image (they are flat)
        i_out = 10
    
        # 4 hidden layer network!
        weights = ((784, 512, NENonlinear('sigmoid')), 
                   (512, 256, NENonlinear('sigmoid')),
                   (256, 16, NENonlinear('sigmoid')),
                   (16, 10, NENonlinear('sigmoid')))
    
        # initialise network
        print("initialising network...")
        #nn = NeuralNetwork(i_input, i_hidden, i_out, Nonlinear('sigmoid'), False, 0.1)
        nnn = NENNN(inputs=i_input, 
                    weights=weights, 
                    outputs=i_out,
                    alpha=0.01)
        print("  network initialised!")
    
    # train networkn
    loops = 100
    print("training network for {} loops".format(loops))
    nnn.train(x, y, loops)
    
    save = input("save network? (y/N): ")
    if save.lower() == 'y':
        fname = input("save network as: ")
        with open(fname, 'wb+') as f:
            pickle.dump(nnn, f)
```

So here is the file in full
```python
#!/usr/bin/python

# Schlerp's neural network
# ----------------------
#
# a bunch of shit for me to fuck around with and learn 
# neural networks.
# currently idea is:
#  * pick a non linear
#  
#  * initialise a network with the amount of input,
#    hidden, and output nodes that you want for the 
#    data
#  
#  * train the network with training x and y
#  
#  * test it using nn.guess(x) with a known y
#  
#  * if happy, test with real world data :P
#


# example 6 layer nn for solving mnist
# i  -h   -h   -h   -h   -h  -o
# 784-2500-2000-1500-1000-500-10

import numpy as np
import os

class Nonlinear(object):
    """
    Nonlinear
    ---------
    this is used to set up a non linear for a
    network. The idea is you can instantiate it 
    and set what type of non linear function it 
    will be for that particular neaural network
    """
    
    _FUNC_TYPES = ('sigmoid',
                   'softmax',
                   'relu',
                   'tanh',
                   'softplus')
    
    def __init__(self, func_type='sigmoid'):
        if func_type in self._FUNC_TYPES:
            if func_type == self._FUNC_TYPES[0]:
                # sigmoid
                self._FUNCTION = self._FUNC_TYPES[0]
            elif func_type == self._FUNC_TYPES[1]:
                # softmax
                self._FUNCTION = self._FUNC_TYPES[1]
            elif func_type == self._FUNC_TYPES[2]:
                # relu
                self._FUNCTION = self._FUNC_TYPES[2]
            elif func_type == self._FUNC_TYPES[3]:
                # tanh
                self._FUNCTION = self._FUNC_TYPES[3]
            elif func_type == self._FUNC_TYPES[4]:
                # tanh
                self._FUNCTION = self._FUNC_TYPES[4]
        else:
            # default to sigmoid on invalid choice?
            print("incorrect option `{}`".format(func_type))
            print("defaulting to sigmoid")
            self._init_sigmoid()
    
    def __call__(self, x, derivative=False):
        ret = None
        if self._FUNCTION == self._FUNC_TYPES[0]:
            # sigmoid
            if derivative:
                ret = x*(1-x)
            else:
                try:
                    ret = 1/(1+np.exp(-x))
                except:
                    ret = 0.0
        elif self._FUNCTION == self._FUNC_TYPES[1]:
            # softmax
            if derivative:
                # from below + http://www.derivative-calculator.net/
                ret = 2*(np.exp(x) / np.sum(np.exp(x)))
            else:
                # from: https://gist.github.com/stober/1946926
                #e_x = np.exp(x - np.max(x))
                #ret = e_x / e_x.sum()
                # from: http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
                ret = np.exp(x) / np.sum(np.exp(x))
        elif self._FUNCTION == self._FUNC_TYPES[2]:
            # relu
            if derivative:
                # from below + http://www.derivative-calculator.net/
                ret = 2*(abs(x))
            else:
                ret = x*(abs(x))
        elif self._FUNCTION == self._FUNC_TYPES[3]:
            # tanh
            if derivative:
                # from my own memory of calculus :P
                ret = 1.0-x**2
            else:
                ret = np.tanh(x)
        elif self._FUNCTION == self._FUNC_TYPES[3]:
            # softmax
            if derivative:
                # from wikipedia
                ret = 1.0/(1+np.exp(-x))
            else:
                ret = np.log(1+np.exp(x))
        return ret

class NeuralNetwork(object):
    """
    Neural network
    --------------
    This is my neural netowrk class, it basically holds all 
    my variables and uses my other functions/classes
    """
    def __init__(self, input, hidden, output, non_lin=Nonlinear(), bias=False, alpha=1, ):
        if bias:
            self._BIAS = True
            self._INPUT = input + 1
        else:
            self._BIAS = False
            self._INPUT = input
        self._ALPHA = alpha
        self._HIDDEN = hidden
        self._OUTPUT = output
        self.non_lin = non_lin
        self._init_nodes()

    def _init_nodes(self):
        # set up weights (synapses)
        self.w_in = np.random.randn(self._INPUT, self._HIDDEN) 
        self.w_out = np.random.randn(self._HIDDEN, self._OUTPUT)
        # set up changes
        #self.change_in = np.zeros((self._INPUT, self._HIDDEN))
        #self.change_out = np.zeros((self._HIDDEN, self._OUTPUT))        
        
    def _do_layer(self, layer_in, weights):
        """Does the actual calcs between layers :)"""
        ret = self.non_lin(np.dot(layer_in, weights))
        return ret
    
    #def _error_delta(self, layer_in, y):
        #layer_error = y - layer_in
        #layer_delta = layer_error * self.non_lin(derivative=True)
        #return layer_error, layer_delta
        
    def train(self, x, y, train_loops=1000):
        for i in range(train_loops):

            # from: https://iamtrask.github.io/2015/07/28/dropout/
            
            # Why Dropout: Dropout helps prevent weights from converging to 
            # identical positions. It does this by randomly turning nodes off 
            # when forward propagating. It then back-propagates with all the 
            # nodes turned on.
            # A good initial configuration for this for hidden layers is 50%. 
            # If applying dropout to an input layer, it's best to not exceed 
            # 25%.
            # use Dropout during training. Do not use it at runtime or on your 
            # testing dataset.
            
            #if do_dropout:
                #layer_1 *= np.random.binomial([np.ones((len(X),hidden_dim))],
                                          #1-dropout_percent)[0] * \
                                          #(1.0/(1-dropout_percent))
            
            # set up layers
            layer0 = x
            layer1 = self._do_layer(layer0, 
                                    self.w_in)
            layer2 = self._do_layer(layer1,
                                    self.w_out)
            
            # calculate errors
            layer2_error = y - layer2
            layer2_delta = layer2_error * self.non_lin(layer2, derivative=True)
            
            layer1_error = layer2_delta.dot(self.w_out.T)
            layer1_delta = layer1_error * self.non_lin(layer1, derivative=True)
            
            if (i % (train_loops/10)) == 0:
                print("loop: {}".format(i))
                print("Layer1 Error: {}".format(np.mean(np.abs(layer1_error))))                
                print("Layer2 Error: {}".format(np.mean(np.abs(layer2_error))))
                print("Guess: ")
                print(layer2[0])               
                print("Guess (round): ")
                print(np.round(layer2[0], 1))
                print("Actual: ")
                print(y[0])
                
            #if (i % (train_loops/100)) == 0:
                #print("currently on loop: {}".format(i))
            # backpropagate error
            self.w_out += self._ALPHA * layer1.T.dot(layer2_delta)
            self.w_in += self._ALPHA * layer0.T.dot(layer1_delta)
            
    
    def guess(self, x):
        _in = x
        _hidden = self.non_lin(np.dot(_in, self.w_in))
        _out = self.non_lin(np.dot(_hidden, self.w_out))
        return _out


class NNN(object):
    """N-layered neural network"""
    def __init__(self, inputs, weights, outputs, alpha):
        self.inputs = inputs
        self.outputs = outputs
        self._ALPHA = alpha
        self._num_of_weights = len(weights)
        self._LAYER_DEFS = {}
        self.WEIGHT_DATA = {}
        self.LAYER_FUNC = {}
        self.LAYERS = {}
        for i in range(self._num_of_weights):
            #(in, out, nonlin)
            self._LAYER_DEFS[i] = {'in': weights[i][0],
                              'out': weights[i][1],
                              'nonlin': weights[i][2]}
        print(self._LAYER_DEFS)
        self._init_layers()
    
    def _init_layers(self):
        for i in range(self._num_of_weights):
            _in = self._LAYER_DEFS[i]['in']
            _out = self._LAYER_DEFS[i]['out']
            _nonlin = self._LAYER_DEFS[i]['nonlin']
            self.WEIGHT_DATA[i] = np.random.randn(_in, _out)
            self.LAYER_FUNC[i] = _nonlin
    
    def _do_layer(self, prev_layer, next_layer, nonlin):
        """Does the actual calcs between layers :)"""
        ret = nonlin(np.dot(prev_layer, next_layer))
        return ret

    def train(self, x, y, train_loops=100):
        for j in range(train_loops):
            # set up layers
            prev_layer = x
            prev_y = y
            next_weight = None
            l = 0
            self.LAYERS[l] = x
            for i in range(self._num_of_weights):
                l += 1
                next_weight = self.WEIGHT_DATA[i]
                nonlin = self.LAYER_FUNC[i]
                current_layer = self._do_layer(prev_layer, next_weight, nonlin)
                self.LAYERS[l] = current_layer
                prev_layer = current_layer
            last_layer = current_layer
            #print(last_layer)
            #
            #layer2_error = y - layer2
            #layer2_delta = layer2_error * self.non_lin(layer2, derivative=True)
            
            #layer1_error = layer2_delta.dot(self.w_out.T)
            #layer1_delta = layer1_error * self.non_lin(layer1, derivative=True)
            
            #self.w_out += self._ALPHA * layer1.T.dot(layer2_delta)
            #self.w_in += self._ALPHA * layer0.T.dot(layer1_delta)              
            
            # calculate errors
            output_error = y - last_layer
            output_nonlin = self.LAYER_FUNC[self._num_of_weights - 1]
            output_delta = output_error * output_nonlin(last_layer, derivative=True)

            prev_delta = output_delta
            prev_layer = last_layer
            for i in reversed(range(self._num_of_weights)):
                weight = self.WEIGHT_DATA[i]
                current_weight_error = prev_delta.dot(weight.T)
                current_weight_nonlin = self.LAYER_FUNC[i]
                current_weight_delta = current_weight_error * current_weight_nonlin(self.LAYERS[i], derivative=True)
                # backpropagate error
                self.WEIGHT_DATA[i] += self._ALPHA * self.LAYERS[i].T.dot(prev_delta)
                prev_delta = current_weight_delta
                

            if (j % (train_loops/10)) == 0:
                print("loop: {}".format(j))
                #print("Layer1 Error: {}".format(np.mean(np.abs(layer1_error))))                
                #print("Layer2 Error: {}".format(np.mean(np.abs(layer2_error))))
                #print("Guess: ")
                #print(last_layer[0])
                #print("output delta: ")
                #print(np.round(output_delta, 2))
                print("Guess (rounded): ")
                print(np.round(last_layer[0], 1))
                print("Actual: ")
                print(y[0])
        
    def guess(self, x):
        prev_layer = x
        prev_y = y
        next_weight = None
        l = 0
        self.LAYERS[l] = x
        for i in range(self._num_of_weights):
            l += 1
            next_weight = self.WEIGHT_DATA[i]
            nonlin = self.LAYER_FUNC[i]
            current_layer = self._do_layer(prev_layer, next_weight, nonlin)
            self.LAYERS[l] = current_layer
            prev_layer = current_layer
        last_layer = current_layer
        return last_layer
            
if __name__ == '__main__':
    import mnist
    try:
        import cPickle as pickle
    except:
        import pickle


    # get data
    if input("load mnist training data?").lower() == 'y':
        load_d = input("  enter filename (eg. 500 = tx-500, ty-500): ")
        with open("mnist/tx{}".format(load_d), 'rb') as f:
            x = pickle.load(f)
        with open("mnist/ty{}".format(load_d), 'rb') as f:
            y = pickle.load(f)
    else:
        print("grabbing data...")
        t_in, t_out = mnist.get_flat_mnist(items=1000, normalize=True)
        print("  got nmist array!")
        print('  {}x{}'.format(len(t_in), len(t_in[0])))
        x = np.array(t_in, dtype=np.float)
        y = np.array(t_out, dtype=np.float)        
    
    
    load = input("load network? (y/N): ")
    if load.lower() == 'y':
        fname = input("network filename: ")
        with open(fname, 'rb') as f:
            nnn = pickle.load(f)
    else:
        # set hypervariables
        i_input = 784 # this is how many pixel per image (they are flat)
        i_out = 10
    
        # 4 hidden layer network!
        weights = ((784, 512, NENonlinear('sigmoid')), 
                   (512, 256, NENonlinear('sigmoid')),
                   (256, 16, NENonlinear('sigmoid')),
                   (16, 10, NENonlinear('sigmoid')))
    
        
        # initialise network
        print("initialising network...")
        #nn = NeuralNetwork(i_input, i_hidden, i_out, Nonlinear('sigmoid'), False, 0.1)
        nnn = NENNN(inputs=i_input, 
                    weights=weights, 
                    outputs=i_out,
                    alpha=0.01)
        print("  network initialised!")
    
    # train networkn
    loops = 100
    print("training network for {} loops".format(loops))
    nnn.train(x, y, loops)
    
    save = input("save network? (y/N): ")
    if save.lower() == 'y':
        fname = input("save network as: ")
        with open(fname, 'wb+') as f:
            pickle.dump(nnn, f)
```

Have a play and tell me what you think!

(also, look for the second post when we implement this using numexpr, a C library that can speed up some of these operations quite a bit!)
