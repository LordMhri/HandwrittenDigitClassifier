#ifndef TRAINER_H
#define TRAINER_H

#include "network.h"

typedef struct  {
    Network *network;

    //hyperparams for the training
    double learning_rate;
    int epochs;
    int batch_size;

    int dataset_size;

    //an array of arrays(the input data)
    double **train_data;

    //an array of the actual answers for the numbers
    uint8_t *train_labels;

    //Accumulate gradients for each weight and bias
    double **acc_grad_w;
    double **acc_grad_b;

    // Reusable buffer for deltas during backpropagation
    double **deltas;

} Trainer;

Trainer* trainer_init(Network *network,double learning_rate,int epochs,int batch_size,
                      int dataset_size,double **train_data,uint8_t* train_labels);


void backpropagation(Trainer *trainer, uint8_t actual_label, double *inputs);

void forward_propagation(Network *network,double *input);

void trainer_free(Trainer *trainer);

void trainer_train(Trainer *trainer);

void apply_gradients(Trainer *trainer, uint32_t batch_size);

#endif