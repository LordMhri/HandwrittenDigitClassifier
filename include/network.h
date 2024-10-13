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

Network* network_init(Network* net,int neurons_input,int neurons_hidden,int neurons_output);
void network_free(Network *net);
void *network_predict(Network *net,double *inputs);


typedef struct Trainer
{
    double *grad_hidden;
    double *grad_output;
} Trainer;

Trainer* trainer_init(Trainer *trainer,Network *net);
void trainer_train(Trainer* trainer, Network* network, double* input, double* output, double lr);
void trainer_free(Trainer* trainer);


#endif