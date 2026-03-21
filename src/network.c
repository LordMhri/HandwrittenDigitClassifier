#include "../include/network.h"
#include "../include/utils.h"
#include <stdlib.h>
#include <stdio.h>




int network_init(Network *network, int num_layers, int *layer_sizes) { 

    network->num_layers = num_layers;
    network->layer_sizes = malloc(num_layers * sizeof(int));
    if (!network->layer_sizes) {
        perror("Error allocating memory for layer_sizes");
        return -1;
    }

    for (int i = 0; i < num_layers; i++) {
        network->layer_sizes[i] = layer_sizes[i];
    }

    // Allocate space for neurons, weights, and biases arrays
    // num_layers - 1 represents the transitions between layers
    network->neurons = calloc(num_layers - 1, sizeof(double *));
    network->weights = calloc(num_layers - 1, sizeof(double *));
    network->biases = calloc(num_layers - 1, sizeof(double *));

    if (!network->neurons || !network->weights || !network->biases) {
        perror("Error allocating memory for network arrays");
        return -1;
    }

    for (int i = 0; i < num_layers - 1; i++) {
        int input_size = network->layer_sizes[i];
        int output_size = network->layer_sizes[i+1];

        // Allocate each layer's neurons (starting from first hidden)
        network->neurons[i] = calloc(output_size, sizeof(double));
        if (!network->neurons[i]) {
            perror("Error allocating neurons");
            return -1;
        }

        // Allocate weights connecting current layer to next
        network->weights[i] = calloc(input_size * output_size, sizeof(double));
        if (!network->weights[i]) {
            perror("Error allocating weights");
            return -1;
        }

        // Allocate biases for next layer
        network->biases[i] = calloc(output_size, sizeof(double));
        if (!network->biases[i]) {
            perror("Error allocating biases");
            return -1;
        }

        // Initialize weights
        he_init(network->weights[i], output_size, input_size);
    }

    return 0;
}

//This is the final part of the network where we
//have the probabilites in the output_neurons
//we just have to pick the maximum one from them
// and have the index of that as our class
uint8_t network_predict(Network *network) {
    uint8_t predicted = 0; 
    
    // The last layer of neurons is our output layer
    double *output_layer = network->neurons[network->num_layers - 2];
    int output_size = network->layer_sizes[network->num_layers - 1];
    
    double curr_max = output_layer[0];

    for (int i = 0; i < output_size; ++i) {
        if (output_layer[i] > curr_max){
            curr_max = output_layer[i];
            predicted = i;
        }
    }

    return predicted;
}


//freeing the memory when done with the program
void network_free(Network *network) {
    if (network->layer_sizes) free(network->layer_sizes);

    for (int i = 0; i < network->num_layers - 1; i++) {
        if (network->neurons[i]) free(network->neurons[i]);
        if (network->weights[i]) free(network->weights[i]);
        if (network->biases[i]) free(network->biases[i]);
    }

    if (network->neurons) free(network->neurons);
    if (network->weights) free(network->weights);
    if (network->biases) free(network->biases);
}
