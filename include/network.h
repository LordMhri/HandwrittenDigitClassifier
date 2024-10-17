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
void network_predict(Network *net,uint8_t **inputs);


typedef struct Trainer
{
    double *grad_hidden;
    double *grad_output;
} Trainer;

Trainer* trainer_init(Trainer *trainer,Network *net);
void trainer_free(Trainer *trainer);
void trainer_SGD_train(Trainer* trainer, Network* network, double* input, double* output, double lr);
void trainer_Mini_Batch_train(Trainer *trainer, Network *network, uint8_t **input, uint8_t *output, uint8_t epoch, uint32_t batch_size, double learning_rate, uint32_t  dataset_size);
//too many parameters , will do for now
#endif