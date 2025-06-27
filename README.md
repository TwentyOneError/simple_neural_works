# Simple workS with neural networks #

## What is this? ##
The module allows working with simple neural networks (Currently, the simplest model of a multilayer perceptron neural network with the backpropagation method and the Leaky ReLu activation function is used).

## Quick Guide ##
The module is based on the following structure:

    
    from simple_neural_works import Neuro
    a = [5,30,30,20,9]
    ne = Neuro(a,0.1)
    ne.fill(False)
    in = [0.5,1,1,0,0.001] # some values to input
    a = ne.get_result(in,False)
    ou = [1,1,1,1,0,1,0,0,1] # some values to train
    ne.backpropagation(ou,False,L1=True)
    ne.save_m("filename.res",False)
    
    

Which Python provides by standard.


----------


### Using ###


Using the library is as simple and convenient as possible:

First, import main module using "from simple_neural_works import Neuro"

The second, you need to load or create an array with data using the load() or fill() function.

Examples of all operations:

Creating an instance of a neural network with which you will then work.

    ne = Neuro(array_width_of_slices_neaural_network,speed of backpropagation)


Filling the weights with initial random values is used to create a neural network from scratch:

    ne.fill(mute)


To load previously saved values from a file:

    ne.load("filenamehere.txt",mute)
    

Function used to obtain the result of a neural network calculation:

    ne.get_result([some float or int values to input in array],mute)


To train a neural network, use the following function. The input is an array, which should be the output of the neural network, the learning rate is controlled by the internal variable `ne.spd`, set manually and during network initialization. You can enable L1 or L2 regularization. Only used after the `get_result()` or `image_get_result()` function.

    ne.backpropagation([some int or float values to train in array],mute,L1,L2)


To quickly save an array of weights:

    ne.save("filenamehere.txt",mute)


Blank for recognizing monochrome numbers. To read data from a PNG image (the image is inverted in color, that is, you need to draw it black, although this is not so important). To avoid specifying the entire path to the file, start the file name with "./".

    ne.image_get_result("filename.png",mute)


----------


### Structure ###

Here are the main variables used in the library; for details, please refer directly to the library code, everything is described there in great detail.

An array storing weight values:
    
    ne.w

An array storing the output values of the activation function:

    ne.ou

Speed of backpropagation (between 0 and 1). If you don’t want to bother, then set it to 0.1. In more detail, at first you can use 1, towards the end of training 0.1:

    ne.speed

Error correction array. The last layer can be used for RNN neural networks.(It can be sent directly to the input of the backpropagation method).

    ne.correction_array

Average squared accuracy of the neural network response (after backpropagation).

    ne.accuracy

Please do not change the width array while working, this will cause the operation to malfunction.
----------

----------

## Developer ##

My GitHub: [link](https://github.com/TwentyOneError)

My Email: ourmail20210422@gmail.com

----------

I would be glad if someone knowledgeable about the topic gives advice or points out errors.

----------

Русский гайд будет позже, я так знатно подзаколебался писать гайд на английском на никому ненужный кусок говна написанный на коленке.

![img.png](https://aif-s3.aif.ru/images/018/907/27e9d88db6e449ff7b17a8f6c890f776.jpg)

### Я устал. Я сделал все что мог. ###
