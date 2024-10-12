#include "../include/network.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double ReLU(double x) {
    return x > 0 ? x: 0;
}

double ReLU_prime(double x){
    return x > 0 ? 1 : 0;
}


Network *network_init(Network *net,int neurons_input,int neurons_output,int neurons_hidden){
    net->neurons_input = neurons_input;
    net->neurons_hidden = neurons_hidden;
    net->neurons_output = neurons_output;


    //allocate memory
    net->weights_hidden = calloc(neurons_input * neurons_hidden,sizeof(*net->weights_hidden));
    net->weights_output = calloc(neurons_output * neurons_hidden,sizeof(*net->weights_output));
    net->bias_hidden = calloc(neurons_hidden,sizeof(*net->bias_hidden));
    net->bias_output = calloc(neurons_output,sizeof(*net->bias_output));
    net->hiddenNeuron = calloc(neurons_hidden , sizeof(*net->hiddenNeuron));
    net->outputNeuron = calloc(neurons_output,sizeof(*net->hiddenNeuron));

    //initialize weights
    for (int i = 0; i < net->neurons_hidden; i++)
    {
        net->weights_hidden[i] = (double) rand() / RAND_MAX;
    }
    for (int i = 0; i < net->neurons_output; i++)
    {
        net->weights_output[i] = (double) rand() / RAND_MAX;
    }

    return net;
}

void network_free(Network *net){
    free(net->bias_hidden);
    free(net->weights_output);
    free(net->weights_hidden);
    free(net->bias_hidden);
    free(net->hiddenNeuron);
    free(net->outputNeuron);    
}

void *network_predict(Network *net,double *inputs) {
    for (int i = 0; i < net->neurons_hidden; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < net->neurons_input; j++)
        {
            sum += inputs[j] * net->weights_hidden[j*net->neurons_hidden + i];
        }
        net->hiddenNeuron[i] = ReLU(sum+net->bias_hidden[i]);
    }

    for (int i = 0; i < net->neurons_output; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < net->neurons_hidden; i++)
        {
            sum += inputs[j] * net->weights_output[j*net->neurons_output + i];
        }
        net->outputNeuron[i] = ReLU(sum + net->bias_output[i]);
    }
    
    
}