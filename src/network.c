#include "../include/network.h"
#include "../include/utils.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <openblas/cblas.h>



int network_init(Network *network, int input_neurons_num,int first_hidden_num,int second_hidden_num,int output_neurons_num) { 

    //Specifying the number of neurons in each layer
    network->neurons_input = input_neurons_num;
    network->first_hidden_neuron_num = first_hidden_num;
    network->second_hidden_neuron_num = second_hidden_num;
    network->output_neurons_num = output_neurons_num;


    //allocate space for the biases in the hidden layers
    network->input_to_first_bias = calloc(first_hidden_num ,sizeof(double));
    if(!network->input_to_first_bias) {
        perror("Error allocating memory for  input to first bias");
        return -1; // -1 for error
    }

    
    network->first_to_second_bias = calloc(second_hidden_num ,sizeof(double));
    if(!network->first_to_second_bias) {
        perror("Error allocating memory for first to second bias");
        free(network->input_to_first_bias);
        return -1; // -1 for error
    }

    network->second_to_output_bias = calloc(output_neurons_num   ,sizeof(double));
    if(!network->second_to_output_bias) {
        perror("Error allocating memory for second to final bias");
        free(network->input_to_first_bias);
        free(network->first_to_second_bias);
        return -1; // -1 for error
    }
    

    //allocate space for the weights
    network->input_to_first_weight = calloc(input_neurons_num * first_hidden_num ,sizeof(double));
    if(!network->input_to_first_weight) {
        perror("Error allocating memory for input to first weight");
        free(network->input_to_first_bias);
        free(network->first_to_second_bias);
        free(network->second_to_output_bias);
        return -1; // -1 for error
    }


    network->first_to_second_weight = calloc(first_hidden_num * second_hidden_num ,sizeof(double));
    if(!network->first_to_second_weight) {
        perror("Error allocating memory for first to second weight");
        free(network->input_to_first_bias);
        free(network->first_to_second_bias);
        free(network->second_to_output_bias);
        free(network->input_to_first_weight);

        return -1; // -1 for error
    }



    network->second_to_output_weight = calloc(second_hidden_num * output_neurons_num  ,sizeof(double));
    if(!network->second_to_output_weight) {
        perror("Error allocating memory for second to final weight");
        free(network->input_to_first_bias);
        free(network->first_to_second_bias);
        free(network->second_to_output_bias);
        free(network->input_to_first_weight);
        free(network->first_to_second_weight);
        return -1; // -1 for error
    }


    //allocate space fot the neurons
    //TODO : freeing memory if any one of the below callocs fail
    network->first_hidden_neurons = calloc(first_hidden_num,sizeof(double));
    network->second_hidden_neurons = calloc(second_hidden_num,sizeof(double));
    network->output_neurons = calloc(output_neurons_num,sizeof(double));

    he_init(network->input_to_first_weight,input_neurons_num,first_hidden_num);
    he_init(network->first_to_second_weight,first_hidden_num,second_hidden_num);
    he_init(network->second_to_output_weight,second_hidden_num,output_neurons_num);

    return 0;
}

//This is the feedforward part of the network
void network_predict(Network *network, double *inputs) {
    // From input to first hidden layer
    matrix_multiply(network->first_hidden_neurons, 
                    inputs, 
                    network->input_to_first_weight, 
                    1, 
                    network->neurons_input, 
                    network->first_hidden_neuron_num);

    // Add biases to the first hidden layer and apply ReLU activation
    for (int i = 0; i < network->first_hidden_neuron_num; i++) {
        network->first_hidden_neurons[i] += network->input_to_first_bias[i];
        network->first_hidden_neurons[i] = ReLU(network->first_hidden_neurons[i]);
    }

    // From first hidden layer to second hidden layer
    matrix_multiply(network->second_hidden_neurons, 
                    network->first_hidden_neurons, 
                    network->first_to_second_weight, 
                    1, 
                    network->first_hidden_neuron_num, 
                    network->second_hidden_neuron_num);

    // Add biases to the second hidden layer and apply ReLU activation
    for (int i = 0; i < network->second_hidden_neuron_num; i++) {
        network->second_hidden_neurons[i] += network->first_to_second_bias[i];
        network->second_hidden_neurons[i] = ReLU(network->second_hidden_neurons[i]);
    }

    // From second hidden layer to output layer
    matrix_multiply(network->output_neurons, 
                    network->second_hidden_neurons, 
                    network->second_to_output_weight, 
                    1, 
                    network->second_hidden_neuron_num, 
                    network->output_neurons_num);

    // Add biases to the output layer
    for (int i = 0; i < network->output_neurons_num; i++) {
        network->output_neurons[i] += network->second_to_output_bias[i];
    }

    // Apply softmax activation to the output layer
    softmax(network->output_neurons, network->output_neurons_num);
}


//freeing the memory when done with the program
void network_free(Network *network) {
    free(network->input_to_first_bias);
    free(network->input_to_first_weight);
    free(network->first_to_second_bias);
    free(network->first_to_second_weight);
    free(network->second_to_output_bias);
    free(network->second_to_output_weight);
    free(network->first_hidden_neurons);
    free(network->second_hidden_neurons);
    free(network->output_neurons);
}

