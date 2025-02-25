#include "../include/trainer.h"
#include "../include/utils.h"
#include "../include/network.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Trainer *trainer_init(Trainer *trainer,Network *network) {


    trainer->acc_grad_bias_hidden = calloc(network->neurons_hidden, sizeof(double));
    trainer->acc_grad_weights_hidden = calloc(network->neurons_input * network->neurons_hidden, sizeof(double));
    trainer->acc_grad_weights_output = calloc(network->neurons_hidden * network->neurons_output, sizeof(double));
    trainer->acc_grad_bias_output = calloc(network->neurons_output, sizeof(double));


    return trainer;
}




double ReLU_Prime(double x) {
    return x > 0 ? 1 : 0;
}
void clip_gradients(double *gradients, int size, double threshold){
    for (int i = 0; i < size; i++)
    {
        if (gradients[i] > threshold)
        {
            gradients[i] = threshold;
        }
        else if (gradients[i] < -threshold)
        {
            gradients[i] = -threshold;
        }
    }
}
void backpropagation(Network *network, uint8_t actual_label, Trainer *trainer, double *inputs) { // Modified: take single actual_label
    // Convert label to one-hot vector (for cross-entropy loss)
    double target_output[10] = {0.0}; // Assuming 10 output neurons for digits 0-9
    target_output[actual_label] = 1.0;

    // Calculate output layer gradients (delta_output)
    double grad_output[10];
    for (int i = 0; i < network->neurons_output; i++) {
        grad_output[i] = network->outputNeuron[i] - target_output[i];
        // printf("Output Neuron %d, Output Act: %.4f, Target: %.4f, Grad: %.4f\n", i, network->outputNeuron[i], target_output[i], grad_output[i]); // Debug print
    }
    clip_gradients(grad_output, network->neurons_output, 1.0);
    // printf("Clipped Output Gradients: [%.4f, %.4f, ...]\n", grad_output[0], grad_output[1]); // Debug print

    // Calculate hidden layer gradients (delta_hidden)
    double grad_hidden[network->neurons_hidden];
    for (int i = 0; i < network->neurons_hidden; i++) {
        double sum = 0.0;
        for (int j = 0; j < network->neurons_output; j++) {
            sum += grad_output[j] * network->weights_output[i * network->neurons_output + j];
        }
        grad_hidden[i] = sum * ReLU_Prime(network->hiddenNeuron[i]);
        // printf("Hidden Neuron %d, Act: %.4f, ReLU': %.4f, Sum Weighted Output Grads: %.4f, Grad: %.4f\n", i, network->hiddenNeuron[i], ReLU_Prime(network->hiddenNeuron[i]), sum, grad_hidden[i]); // Debug print
    }
    clip_gradients(grad_hidden, network->neurons_hidden, 1.0);
    // printf("Clipped Hidden Gradients: [%.4f, %.4f, ...]\n", grad_hidden[0], grad_hidden[1]); // Debug print


    // Accumulate gradients for output layer weights and biases
    for (int i = 0; i < network->neurons_hidden; i++) {
        for (int j = 0; j < network->neurons_output; j++) {
            trainer->acc_grad_weights_output[i * network->neurons_output + j] += grad_output[j] * network->hiddenNeuron[i];
            // printf("Accumulating Output Weight Grad [%d][%d]: += %.4f (delta_out[j] * hiddenNeuron[i])\n", i, j, grad_output[j] * network->hiddenNeuron[i]); // Debug print
        }
    }

    for (int i = 0; i < network->neurons_output; i++) {
        trainer->acc_grad_bias_output[i] += grad_output[i];
        // printf("Accumulating Output Bias Grad [%d]: += %.4f (delta_out[i])\n", i, grad_output[i]); // Debug print
    }

    // Accumulate gradients for hidden layer weights and biases
    for (int i = 0; i < network->neurons_input; i++) {
        for (int j = 0; j < network->neurons_hidden; j++) {
            trainer->acc_grad_weights_hidden[i * network->neurons_hidden + j] += grad_hidden[j] * inputs[i];
            // printf("Accumulating Hidden Weight Grad [%d][%d]: += %.4f (delta_hidden[j] * inputs[i])\n", i, j, grad_hidden[j] * inputs[i]); // Debug print
        }
    }

    for (int i = 0; i < network->neurons_hidden; i++) {
        trainer->acc_grad_bias_hidden[i] += grad_hidden[i];
        // printf("Accumulating Hidden Bias Grad [%d]: += %.4f (delta_hidden[i])\n", i, grad_hidden[i]); // Debug print
    }
};


void apply_gradients(Trainer *trainer, Network *network, double learning_rate, uint32_t batch_size) {
    // Update weights for the output layer
    for (int i = 0; i < network->neurons_hidden; i++)
    {
        for (int j = 0; j < network->neurons_output; j++)
        {
            network->weights_output[i * network->neurons_output + j] -= learning_rate * (trainer->acc_grad_weights_output[i * network->neurons_output + j] / batch_size); // Apply averaged gradient
            trainer->acc_grad_weights_output[i * network->neurons_output + j] = 0.0; // Reset accumulator
        }
    }

    // Update biases for the output layer
    for (int i = 0; i < network->neurons_output; i++)
    {
        network->bias_output[i] -= learning_rate * (trainer->acc_grad_bias_output[i] / batch_size); // Apply averaged gradient
        trainer->acc_grad_bias_output[i] = 0.0; // Reset accumulator
    }

    // Update weights for the hidden layer
    for (int i = 0; i < network->neurons_input; i++)
    {
        for (int j = 0; j < network->neurons_hidden; j++)
        {
            network->weights_hidden[i * network->neurons_hidden + j] -= learning_rate * (trainer->acc_grad_weights_hidden[i * network->neurons_hidden + j] / batch_size); // Apply averaged gradient
            trainer->acc_grad_weights_hidden[i * network->neurons_hidden + j] = 0.0; // Reset accumulator
        }
    }

    // Update biases for the hidden layer
    for (int i = 0; i < network->neurons_hidden; i++)
    {
        network->bias_hidden[i] -= learning_rate * (trainer->acc_grad_bias_hidden[i] / batch_size); // Apply averaged gradient
        trainer->acc_grad_bias_hidden[i] = 0.0; // Reset accumulator
    }
}

void trainer_Mini_Batch_train(Trainer *trainer, Network *network, double **input, uint8_t *output, uint8_t epoch, uint32_t batch_size, double learning_rate, uint32_t  dataset_size)
{
    for (uint8_t e = 0; e < epoch; e++)
    {
        shuffle(input, output, dataset_size);

        uint32_t num_of_batches = (uint32_t)dataset_size / batch_size;
        double total_loss = 0.0; // Initialize total loss for the epoch

        for (uint32_t n = 0; n < num_of_batches; n++)
        {
            double **inputBatch2D = get_batch_2D(input, batch_size, n);

            for (uint32_t b = 0; b < batch_size; b++)
            {
                double *inputBatch = flatten_2D(inputBatch2D, b);
                network_predict(network, inputBatch);


                double loss = cross_entropy_loss(network->outputNeuron, output[n * batch_size + b]); 

                total_loss += loss;

                backpropagation(network, output[n * batch_size + b], trainer, inputBatch);

                free(inputBatch);
            }

            // Apply gradients after processing the entire batch
            apply_gradients(trainer, network, learning_rate, batch_size);


            // Free each image in the batch
            for (uint32_t b = 0; b < batch_size; b++)
            {
                free(inputBatch2D[b]);
            }
            free(inputBatch2D);
        }

        double average_loss = total_loss / num_of_batches; 
        printf("Epoch %d: Average Loss = %f\n", e + 1, average_loss);
    }
}


void trainer_free(Trainer* trainer){
    free(trainer->acc_grad_weights_hidden);
    free(trainer->acc_grad_bias_hidden);
    free(trainer->acc_grad_weights_output);
    free(trainer->acc_grad_bias_output);
}

