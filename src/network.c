#include "../include/network.h"
#include "../include/utils.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <openblas/cblas.h>



int network_init(Network* network, int neurons_input, int neurons_hidden, int neurons_output) { // Return int to indicate success/failure
    network->neurons_input = neurons_input;
    network->neurons_hidden = neurons_hidden;
    network->neurons_output = neurons_output;

    network->weights_hidden = calloc(neurons_input * neurons_hidden, sizeof(*network->weights_hidden));
    if (!network->weights_hidden) { 
        perror("calloc failed for weights_hidden");
        return -1; 
    }

    network->weights_output = calloc(neurons_hidden * neurons_output, sizeof(*network->weights_output));
    if (!network->weights_output) { 
        perror("calloc failed for weights_output");
        free(network->weights_hidden); // Free previously allocated memory
        return -1; 
    }

    network->bias_hidden = calloc(neurons_hidden, sizeof(*network->bias_hidden));
    if (!network->bias_hidden) { 
        perror("calloc failed for bias_hidden");
        free(network->weights_hidden);
        free(network->weights_output);
        return -1; 
    }

    network->bias_output = calloc(neurons_output, sizeof(*network->bias_output));
    if (!network->bias_output) { 
        perror("calloc failed for bias_output");
        free(network->weights_hidden);
        free(network->weights_output);
        free(network->bias_hidden);
        return -1; 
    }

    network->hiddenNeuron = calloc(neurons_hidden, sizeof(*network->hiddenNeuron));
    if (!network->hiddenNeuron) { 
        perror("calloc failed for hiddenNeuron");
        free(network->weights_hidden);
        free(network->weights_output);
        free(network->bias_hidden);
        free(network->bias_output);
        return -1; 
    }

    network->outputNeuron = calloc(neurons_output, sizeof(*network->outputNeuron));
    if (!network->outputNeuron) {
        perror("calloc failed for outputNeuron");
        free(network->weights_hidden);
        free(network->weights_output);
        free(network->bias_hidden);
        free(network->bias_output);
        free(network->hiddenNeuron);
        return -1; 
    }

    //initialize weights
    he_init(network->weights_hidden, neurons_hidden, neurons_input);
    he_init(network->weights_output, neurons_output, neurons_hidden);
    
    
    return 0;


}


void network_predict(Network *network, double *inputs) {

    // Initialize hiddenNeuron with bias_hidden
    for (int i = 0; i < network->neurons_hidden; i++) {
        network->hiddenNeuron[i] = network->bias_hidden[i]; // Initialize with bias
    }
    cblas_dgemv(CblasRowMajor, CblasNoTrans, network->neurons_hidden, 
    network->neurons_input, 1.0, network->weights_hidden, network->neurons_input, inputs, 1, 1.0, network->hiddenNeuron, 1);

    // Forward pass from input to hidden layer
    for (int i = 0; i < network->neurons_hidden; i++)
    {
        network->hiddenNeuron[i] = ReLU(network->hiddenNeuron[i]);
    }

    // for (int i = 0; i < network->neurons_output; i++) {
    //     printf("sum is %f for neuron %d\n", network->outputNeuron[i], i);
    // }

    //this is matrix vector calculation
    //essentially does y = alpha*A*x + beta*y
    // Initialize outputNeuron with bias_output
    for (int i = 0; i < network->neurons_output; i++) {
        network->outputNeuron[i] = network->bias_output[i]; // Initialize with bias
    }
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

