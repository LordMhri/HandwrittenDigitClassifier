#include "../include/network.h"
#include "../include/utils.h"
#include "../include/trainer.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <cblas.h>

//he_intialization of random weights
void he_init(double *weights,int neurons_output,int neurons_input) {
    double stddev = sqrt(2.0/neurons_input);
    for (int i= 0; i < neurons_input*neurons_output; i++)
    {
        weights[i] = (double) rand() / RAND_MAX * stddev * 2 - stddev;
    }
}

void clip_gradients(double *gradients, int size, double threshold)
{
    double norm = 0.0;
    for (int i = 0; i < size; i++)
    {
        norm += gradients[i] * gradients[i];
    }
    norm = sqrt(norm);

    if (norm > threshold)
    {
        double scale = threshold / norm;
        for (int i = 0; i < size; i++)
        {
            gradients[i] *= scale;
        }
    }
}

double ReLU(double x){
    return x > 0 ? x : 0.01*x;
}

double ReLU_Prime(double x) {
    //experimenting with leakyReLu because of dead neurons
    return x > 0 ? 1 : 0.01;
}

void softmax(double *input, int length) {
    double max_input = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_input) {
            max_input = input[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        input[i] = exp(input[i] - max_input);
        sum += input[i];
    }

    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}


void backpropagation(Network *network, uint8_t *batch_output, Trainer *trainer, double learning_rate, double *inputs) {
    // Calculate output layer gradients
    for (int i = 0; i < network->neurons_output; i++) {
        trainer->grad_output[i] = network->outputNeuron[i] - batch_output[i];
        // printf("Output gradients %f\n", trainer->grad_output[i]);
    }
    clip_gradients(trainer->grad_output, network->neurons_output, 1.0);

    // Calculate hidden layer gradients
    for (int i = 0; i < network->neurons_hidden; i++) {
        double sum = 0.0;
        for (int j = 0; j < network->neurons_output; j++) {
            sum += trainer->grad_output[j] * network->weights_output[i * network->neurons_output + j];
        }
        trainer->grad_hidden[i] = sum * ReLU_Prime(network->hiddenNeuron[i]);
        // printf("hidden gradients %f\n", trainer->grad_hidden[i]);
    }

    clip_gradients(trainer->grad_hidden, network->neurons_hidden, 1.0);

    // Update weights for the output layer
    for (int i = 0; i < network->neurons_hidden; i++) {
        for (int j = 0; j < network->neurons_output; j++) {
            network->weights_output[i * network->neurons_output + j] -= learning_rate * trainer->grad_output[j] * network->hiddenNeuron[i];
        }
    }

    // Update biases for the output layer
    for (int i = 0; i < network->neurons_output; i++) {
        network->bias_output[i] -= learning_rate * trainer->grad_output[i];
    }

    // Update weights for the hidden layer
    for (int i = 0; i < network->neurons_input; i++) {
        for (int j = 0; j < network->neurons_hidden; j++) {
            network->weights_hidden[i * network->neurons_hidden + j] -= learning_rate * trainer->grad_hidden[j] * inputs[i]; 
        }
    }

    // Update biases for the hidden layer
    for (int i = 0; i < network->neurons_hidden; i++) {
        network->bias_hidden[i] -= learning_rate * trainer->grad_hidden[i];
    }
}



void swap_double_ptrs(double **a, double **b) {
    double *temp = *a;
    *a = *b;
    *b = temp;
}


void network_init(Network* network,int neurons_input,int neurons_hidden,int neurons_output) {
    network->neurons_input = neurons_input;
    network->neurons_hidden = neurons_hidden;
    network->neurons_output = neurons_output;

    network->weights_hidden = calloc(neurons_input*neurons_hidden , sizeof(*network->weights_hidden));
    network->weights_output = calloc(neurons_hidden*neurons_output,sizeof(*network->weights_output));
    network->bias_hidden = calloc(neurons_hidden,sizeof(*network->bias_hidden));
    network->bias_output = calloc(neurons_output,sizeof(*network->bias_output));


    network->hiddenNeuron = calloc(neurons_hidden,sizeof(*network->hiddenNeuron));
    network->outputNeuron = calloc(neurons_output,sizeof(*network->outputNeuron));

    //initialize weights
    he_init(network->weights_hidden,neurons_hidden,neurons_input);  // Fix: neurons_hidden, neurons_input
    he_init(network->weights_output,neurons_output,neurons_hidden); // Fix: neurons_output, neurons_hidden



}



void network_predict(Network *network, double *inputs) {

    cblas_dgemv(CblasRowMajor, CblasNoTrans, network->neurons_hidden, 
    network->neurons_input, 1.0, network->weights_hidden, network->neurons_input, inputs, 1, 1.0, network->hiddenNeuron, 1);

    // Forward pass from input to hidden layer
    // Apply activation function (e.g., ReLU)
    for (int i = 0; i < network->neurons_hidden; i++)
    {
        network->hiddenNeuron[i] = ReLU(network->hiddenNeuron[i]);
    }

    // for (int i = 0; i < network->neurons_output; i++) {
    //     printf("sum is %f for neuron %d\n", network->outputNeuron[i], i);
    // }

    //this is matrix vector calculation
    //essentially does y = alpha*A*x + beta*y
    cblas_dgemv(CblasRowMajor, CblasNoTrans, network->neurons_output, network->neurons_hidden, 1.0, network->weights_output, network->neurons_hidden, network->hiddenNeuron, 1, 1.0, network->outputNeuron, 1);


    softmax(network->outputNeuron, network->neurons_output);

    /*
    This block of code is needed to check if there is something wrong with softmax in the output layer,
    sum must always be exactly 1
    */
    // double sum = 0.0;
    // for (int i = 0; i < network->neurons_output; i++)
    // {
    //     sum += network->outputNeuron[i];
    // }

    // printf("this is after softmax, Sum is %f\n", sum);

}



void network_free(Network *network) {
    free(network->weights_hidden);
    free(network->weights_output);
    free(network->bias_output);
    free(network->bias_hidden);
    free(network->hiddenNeuron);
    free(network->outputNeuron);
}

