# MINST Handwritten Digit Classifier
A simple handwritten digit recognizer built in C, implementing a basic 3-layer feedforward neural network for the MNIST dataset. 
This program takes the minst dataset which is in binary form, transforms the data into an array of array of double data type for each image (height,width) and
classifies 28x28 grayscale images of handwritten digits (0-9) using a neural network trained on the MNIST training dataset.
## Features
- **Language**: C
- **Neural Network Architecture**: 3 layers(input,hidden and output) ... for the time being
- **Activation Funcion**: Leaky ReLU
- **Loss Function** : Cross Entropy Loss
- **Optimization** : Gradient descent with mini batches
## Network Architecture
- **Input Layer** : 28*28 / 784 neurons ( for each pixel in an image)
