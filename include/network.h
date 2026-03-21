#include <stdint.h>
#ifndef NETWORK_H
#define NETWORK_H

typedef struct Network
{
    int num_layers;         // Total number of layers (input + hidden + output)
    int *layer_sizes;       // Number of neurons in each layer

    // neurons[i] is the (i+1)-th layer. Input layer (layer 0) is not stored.
    double **neurons;

    // weights[i] connects layer i to layer i+1
    double **weights;
    
    // biases[i] is for layer i+1
    double **biases;

} Network;

int network_init(Network *network, int num_layers, int *layer_sizes);
uint8_t network_predict(Network *net); //value of the predicted number
void network_free(Network *net);



#endif