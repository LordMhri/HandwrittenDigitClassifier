#include "../include/network.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

//he_intialization of random weights
void he_init(double *weights,int neurons_output,int neurons_input) {
    double stddev = sqrt(2.0/neurons_input);
    for (int i = 0; i < neurons_input*neurons_output; i++)
    {
        weights[i] = (double) rand() / RAND_MAX * stddev * 2 - stddev;
    }
}

double ReLU(double x){
    return x > 0 ? x : 0;
}

double ReLU_Prime(double x) {
    return 1 > 0 ? x : 0;
}

void softmax(double *input,int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++)
    {
        input[i] = exp(input[i]);
        sum  += input[i];
    }
    for (int i = 0; i < length; i++)
    {
        input[i] /= sum;
    }
    // class i = e^input[i] / sum of e^input[i] from i = 0 to i
  
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
    he_init(network->weights_hidden,neurons_output,neurons_input);
    he_init(network->weights_output,neurons_output,neurons_input);

    for (int i = 0; i < network->neurons_hidden; i++)
    {
        printf("weight for hidden for %i is %f\n", i,network->weights_hidden[i]);
    }
    for (int i = 0; i < network->neurons_output; i++)
    {
        printf("weight for output for %i is %f\n", i,network->weights_output[i]);
    }
    

}


void network_predict(Network *network,double *inputs) {
    //forward passes from input to hidden neuron
    for (int i = 0; i < network->neurons_hidden; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < network->neurons_input; j++)
        {
            sum += inputs[j] * network->weights_hidden[j*network->neurons_hidden + i];
        }
        network->hiddenNeuron[i] = ReLU(sum + network->bias_hidden[i]);
    }
    

    //forward pass from hidden to output neuron
    for (int i = 0; i < network->neurons_output; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < network->neurons_hidden; j++)
        {
            sum += network->hiddenNeuron[j] * network->weights_output[j*network->neurons_hidden + i];
        }
        network->outputNeuron[i] = sum+network->bias_output[i];        
    }
    
    softmax(network->outputNeuron,network->neurons_output);

}

void network_free(Network *network) {
    free(network->weights_hidden);
    free(network->weights_output);
    free(network->bias_output);
    free(network->bias_hidden);
    free(network->hiddenNeuron);
    free(network->outputNeuron);
}

