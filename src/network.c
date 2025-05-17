#include "../include/network.h"
#include "../include/utils.h"
#include <stdlib.h>
#include <stdio.h>




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

    network->first_hidden_pre_activation_values = calloc(first_hidden_num,sizeof(double));
    network->second_hidden_pre_activation_values = calloc(second_hidden_num,sizeof(double));
    network->output_pre_activation_values = calloc(output_neurons_num,sizeof(double));

    he_init(network->input_to_first_weight,input_neurons_num,first_hidden_num);
    he_init(network->first_to_second_weight,first_hidden_num,second_hidden_num);
    he_init(network->second_to_output_weight,second_hidden_num,output_neurons_num);

    return 0;
}

//This is the final part of the network where we
//have the probabilites in the network->output_neurons
//we just have to pick the maximum one from them
// and have the index of that as our class
uint8_t network_predict(Network *network) {
    uint8_t predicted = 0; //the predicted value will be a number from 0 to 10
    double curr_max = network->output_neurons[0];

    for (int i = 0; i < network->output_neurons_num; ++i) {
        if (network->output_neurons[i] > curr_max){
            curr_max = network->output_neurons[i];
            predicted = i;
        }
    }

    return predicted;
}


//freeing the memory when done with the program
void network_free(Network *network) {
    free(network->first_hidden_pre_activation_values);
    free(network->second_hidden_pre_activation_values);
    free(network->output_pre_activation_values);
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

