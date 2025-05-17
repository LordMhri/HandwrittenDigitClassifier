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
    //a flattened 2d sequence of numbers
    double **train_data;

    //an array of the actual answers for the numbers
    //the values are between 0 and 9 so we could save
    // some memory using uint8_t instead of int
    uint8_t *train_labels;

    //Accumulate gradients for each weight and bias
    //useful in backpropagation
    double *acc_grad_w1;
    double *acc_grad_b1;

    double *acc_grad_w2;
    double *acc_grad_b2;

    double *acc_grad_w3;
    double *acc_grad_b3;



} Trainer;



Trainer* trainer_init(Network *network,double learning_rate,int epochs,int batch_size,
                      int dataset_size,double **train_data,uint8_t* train_labels);


void backpropagate(Trainer *trainer,Network *network,double *inputs,uint8_t label);

void forward_propagation(Network *network,double *input);
void trainer_free(Trainer *trainer);
//void trainer_Mini_Batch_train(Trainer *trainer, Network *network, double **input, uint8_t *output, uint8_t epoch, uint32_t batch_size, double learning_rate, uint32_t  dataset_size);
void apply_gradients(Trainer *trainer, Network *network, double learning_rate, uint32_t batch_size);
#endif