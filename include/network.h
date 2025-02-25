#include <stdint.h>
#ifndef NETWORK_H
#define NETWORK_H

typedef struct Network
{
    double *hiddenNeuron;
    double *outputNeuron;

    double *weights_hidden;
    double *bias_hidden;
    double *weights_output;
    double *bias_output;

    int neurons_hidden;
    int neurons_input;
    int neurons_output;

} Network;

void network_init(Network* net,int neurons_input,int neurons_hidden,int neurons_output);
void network_free(Network *net);
void network_predict(Network *net,double *inputs);



//too many parameters , will do for now
#endif