#include <stdint.h>
#ifndef NETWORK_H
#define NETWORK_H

typedef struct Network
{
    //no array of doubles for the first layer because it is the input layer
    double *first_hidden_neurons;
    double *second_hidden_neurons;
    double *output_neurons;

    //these are the z-values(the preactivated values ) needed for backprop
    double *first_hidden_pre_activation_values;
    double *second_hidden_pre_activation_values;
    double *output_pre_activation_values;

   //input_to_first_transition
   double *input_to_first_weight;
   double *input_to_first_bias;
   
   //first_to_second_transition
   
   double *first_to_second_weight;
   double *first_to_second_bias;
   
   
   //second_to_final_transition
   double *second_to_output_weight;
   double *second_to_output_bias;


    //number of neurons for each layer    
    int second_hidden_neuron_num;
    int first_hidden_neuron_num;    
    int neurons_input;
    int output_neurons_num;

} Network;

int network_init(Network *network, int input_neurons_num,int first_hidden_num,int second_hidden_num,int output_neurons_num);
uint8_t network_predict(Network *net); //value of the predicted number
void network_free(Network *net);



#endif