#ifndef TRAINER_H
#define TRAINER_H

#include "network.h"

typedef struct Trainer
{
    double *grad_hidden;
    double *grad_output;
} Trainer;

Trainer* trainer_init(Trainer *trainer,Network *net);
void trainer_free(Trainer *trainer);
void trainer_SGD_train(Trainer* trainer, Network* network, double* input, double* output, double lr);
void trainer_Mini_Batch_train(Trainer *trainer, Network *network, uint8_t **input, uint8_t *output, uint8_t epoch, uint32_t batch_size, double learning_rate, uint32_t  dataset_size);
int ReLU(double x);
#endif