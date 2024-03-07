# Simple workS with neural networks #

## What is this? ##
The module allows you to work with simple neural networks (At the moment, the simplest convolutional neural network model is used with the method of backpropagation of error and sigmoidal activation function).

## Quick Guide ##
The module is based on the following structure:

    
    ne = Neuro(10,5,5,0.1,False)
    ne.init_m()
    ne.fill_m()
    in = [0.5,1,1,0,0.001] # some values to input
    a = ne.getv(in)
    ou = [1,1,1,1,0,1,0,0,1] # some values to train
    ne.bpn(ou)
    ne.save_m("filename.res")
    
    

Which Python provides by standard.


----------


### Using ###


Using the library is as simple and convenient as possible:

First, import main module using "from simple_neural_works import Neuro"

The second, you need to initialize the creation of an array with data using the init_m function.

Examples of all operations:

Creating an instance of a neural network with which you will then work.

    ne = Neuro(wide, height, depth, speed of backpropagation, Muting the console output)

Initializing an array of the required size before work (required):

    ne.init_m()


Filling the weights with initial random values is used to create a neural network from scratch:

    ne.fill_m()


To load previously saved values from a file:

    ne.load_m("filenamehere.txt")
    

Function used to obtain the result of a neural network calculation:

    ne.getv([some float or int values to input])


To train a neural network, use the following function. The input is an array, which should be the output of the neural network, the learning rate is controlled by the internal variable `ne.spd`, set manually and during network initialization. Only used after the `getv()` or `imgg()` function.

    ne.bpn([some int or float values to train])


To quickly save an array of weights:

    ne.save_m("filename.txt")


For compact (up to two times smaller) but slower saving:

    ne.zip_save_m("filename.txt")


Blank for recognizing monochrome numbers. To read data from a PNG image (the image is inverted in color, that is, you need to draw it black, although this is not so important). To avoid specifying the entire path to the file, start the file name with "./".

    ne.imgg("filename.png")


----------


### Structure ###

Here are the main variables used in the library; for details, please refer directly to the library code, everything is described there in great detail.

An array storing weight values:
    
    ne.w

An array storing the output values of the activation function:

    ne.ou

Variable switching the output of intermediate data (number of operations per execution, execution progress, response accuracy for the `bpn()` function:

    ne.mute

Speed of backpropagation (between 0 and 1). If you don’t want to bother, then set it to 0.1. In more detail, at first you can use 1, towards the end of training 0.1:

    ne.spd

Please do not change the width, height and depth values while working, this will cause the operation to malfunction.
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