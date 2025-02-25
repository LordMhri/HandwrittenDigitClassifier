#include "../include/trainer.h"
#include "../include/utils.h"
#include "../include/network.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Trainer *trainer_init(Trainer *trainer,Network *network) {
    trainer->grad_hidden = calloc(network->neurons_hidden,sizeof(*trainer->grad_hidden));
    trainer->grad_output = calloc(network->neurons_output,sizeof(*trainer->grad_output));
      
    return trainer;
}

double* flatten_2D(double **input, uint32_t index) {
    double* flattened_input = calloc(28 * 28, sizeof(double));
    
    for (uint8_t row = 0; row < 28; row++) {
        for (uint8_t col = 0; col < 28; col++) {
            flattened_input[row * 28 + col] = input[index][row * 28 + col];
            // printf("flattened_input[row * 28 + col] is %f\n",flattened_input[row * 28 + col]);
        }
    }

    return flattened_input;
}


// double cross_entropy_loss(double *output, uint8_t *label, uint32_t batch_size) {
//     double loss = 0.0;
//     for (uint32_t i = 0; i < batch_size; i++)
//     {
//         loss += -log(output[i * 10 + label[i]]);
//     }
//     return loss / batch_size;
// }

void clip_gradients(double *gradients, uint32_t size, double threshold) {
    for (uint32_t i = 0; i < size; i++)
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

int ReLU(double x) {
    return x > 0 ? x : 0;
}

int ReLU_Prime(double x) {
    return x > 0 ? 1 : 0;
}

void backpropagation(Network *network, uint8_t *batch_output, Trainer *trainer, double learning_rate, double *inputs) {
    // Calculate output layer gradients
    for (int i = 0; i < network->neurons_output; i++)
    {
        trainer->grad_output[i] = network->outputNeuron[i] - batch_output[i];
        // printf("Output gradients %f\n", trainer->grad_output[i]);
    }
    clip_gradients(trainer->grad_output, network->neurons_output, 1.0);

    // Calculate hidden layer gradients
    for (int i = 0; i < network->neurons_hidden; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < network->neurons_output; j++)
        {
            sum += trainer->grad_output[j] * network->weights_output[i * network->neurons_output + j];
        }
        trainer->grad_hidden[i] = sum * ReLU_Prime(network->hiddenNeuron[i]);
        // printf("hidden gradients %f\n", trainer->grad_hidden[i]);
    }

    clip_gradients(trainer->grad_hidden, network->neurons_hidden, 1.0);

    // Update weights for the output layer
    for (int i = 0; i < network->neurons_hidden; i++)
    {
        for (int j = 0; j < network->neurons_output; j++)
        {
            network->weights_output[i * network->neurons_output + j] -= learning_rate * trainer->grad_output[j] * network->hiddenNeuron[i];
        }
    }

    // Update biases for the output layer
    for (int i = 0; i < network->neurons_output; i++)
    {
        network->bias_output[i] -= learning_rate * trainer->grad_output[i];
    }

    // Update weights for the hidden layer
    for (int i = 0; i < network->neurons_input; i++)
    {
        for (int j = 0; j < network->neurons_hidden; j++)
        {
            network->weights_hidden[i * network->neurons_hidden + j] -= learning_rate * trainer->grad_hidden[j] * inputs[i];
        }
    }

    // Update biases for the hidden layer
    for (int i = 0; i < network->neurons_hidden; i++)
    {
        network->bias_hidden[i] -= learning_rate * trainer->grad_hidden[i];
    }
};

void trainer_Mini_Batch_train(Trainer *trainer, Network *network, uint8_t **input, uint8_t *output, uint8_t epoch, uint32_t batch_size, double learning_rate, uint32_t dataset_size)
{
    for (uint8_t e = 0; e < epoch; e++)
    {
        shuffle(input, output, dataset_size);

        // Check alignment of a few samples after shuffling
        // printf("Epoch %d: Checking alignment after shuffling...\n", e + 1);
        // for (int i = 0; i < 5; i++)
        // { // Check the first 5 pairs for instance
        //     printf("Sample %d - First pixel of input: %f, Label: %d\n", i, input[i][0], output[i]);
        // }

        uint32_t num_of_batches = (uint32_t)dataset_size / batch_size;
        double total_loss = 0.0; // Initialize total loss for the epoch

        for (uint32_t n = 0; n < num_of_batches; n++)
        {
            double **inputBatch2D = get_batch_2D(input, batch_size, n);

            for (uint32_t b = 0; b < batch_size; b++)
            {
                double *inputBatch = flatten_2D(inputBatch2D, b);
                network_predict(network, inputBatch);

                // Check if the output label aligns with the input
                // printf("Batch %d, Sample %d - Predicted label: %f, Actual label: %d\n", n, b, network->outputNeuron[0], output[b]);

                double loss = cross_entropy_loss(network->outputNeuron, &output[n * batch_size], batch_size);
                total_loss += loss;

                backpropagation(network, &output[b * network->neurons_output], trainer, learning_rate, inputBatch);

                free(inputBatch);
            }

            // Free each image in the batch
            for (uint32_t b = 0; b < batch_size; b++)
            {
                free(inputBatch2D[b]);
            }

            // Free batch array
            free(inputBatch2D);
        }
        // there needs to be some change here
        // we already look divide total_loss by the batch size in the cross entropy loss function
        //so it seems wrong to again use it here
        // average_loss should be total_loss / num of batches not num of batches * batch_size
        double average_loss = total_loss / (num_of_batches * batch_size);
        printf("Epoch %d: Average Loss = %f\n", e + 1, average_loss);
    }
}

void trainer_free(Trainer* trainer){
    free(trainer->grad_hidden);
    free(trainer->grad_output);
}


