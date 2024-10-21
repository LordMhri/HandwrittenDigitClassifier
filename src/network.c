#include "../include/network.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

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
    return x > 0 ? 1 : 0;
}

void softmax(double *input, int length) {
    double max_input = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_input) {
            max_input = input[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        input[i] = exp(input[i] - max_input);
        sum += input[i];
    }

}

double cross_entropy_loss(double *predicted,uint8_t *actual,int num_output) {
    double loss = 0.0;
    for (int i = 0; i < num_output; i++)
    {
        loss -= actual[i] * log(predicted[i] + 1e-9);
    }
    return loss;
}

double** get_batch_2D(double **input, uint32_t batch_size, uint8_t batch_index) {
    // Allocate memory for the batch of images
    double **batch = calloc(batch_size, sizeof(double*));
    // printf("We're called here!");

    for (uint32_t b = 0; b < batch_size; b++) {
        // Allocate memory for each image (28 * 28)
        batch[b] = malloc(28 * 28 * sizeof(double));

        // Copy the image data from the input dataset
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                batch[b][row * 28 + col] = input[batch_index * batch_size + b][row * 28 + col];
                // printf("batch[b][row * 28 + col] is %f\n",batch[b][row * 28 + col] );
            }
        }
    }
    return batch;
}



void backpropagation(Network *network, uint8_t *batch_output, Trainer *trainer, double learning_rate, double *inputs) {
    // Calculate output layer gradients
    for (int i = 0; i < network->neurons_output; i++) {
        trainer->grad_output[i] = network->outputNeuron[i] - batch_output[i];
    }

    // Calculate hidden layer gradients
    for (int i = 0; i < network->neurons_hidden; i++) {
        double sum = 0.0;
        for (int j = 0; j < network->neurons_output; j++) {
            sum += trainer->grad_output[j] * network->weights_output[i * network->neurons_output + j];
        }
        trainer->grad_hidden[i] = sum * ReLU_Prime(network->hiddenNeuron[i]);
    }

    // Update weights for the output layer
    for (int i = 0; i < network->neurons_hidden; i++) {
        for (int j = 0; j < network->neurons_output; j++) {
            network->weights_output[i * network->neurons_output + j] -= learning_rate * trainer->grad_output[j] * network->hiddenNeuron[i];
        }
    }

    // Update biases for the output layer
    for (int i = 0; i < network->neurons_output; i++) {
        network->bias_output[i] -= learning_rate * trainer->grad_output[i];
    }

    // Update weights for the hidden layer
    for (int i = 0; i < network->neurons_input; i++) {
        for (int j = 0; j < network->neurons_hidden; j++) {
            network->weights_hidden[i * network->neurons_hidden + j] -= learning_rate * trainer->grad_hidden[j] * inputs[i]; 
        }
    }

    // Update biases for the hidden layer
    for (int i = 0; i < network->neurons_hidden; i++) {
        network->bias_hidden[i] -= learning_rate * trainer->grad_hidden[i];
    }
}



void swap_double_ptrs(double **a, double **b) {
    double *temp = *a;
    *a = *b;
    *b = temp;
}

//this works now
void shuffle(double **inputs, uint8_t *output, int dataset_size) {
    for (int i = dataset_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        swap_double_ptrs(&inputs[i], &inputs[j]);

        uint8_t temp = output[i];
        // printf("temp is %d\n",temp);
        output[i] = output[j];
        output[j] = temp;
    }
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
    he_init(network->weights_hidden,neurons_hidden,neurons_input);  // Fix: neurons_hidden, neurons_input
    he_init(network->weights_output,neurons_output,neurons_hidden); // Fix: neurons_output, neurons_hidden



}




void network_predict(Network *network, double *inputs) {

    // Forward pass from input to hidden layer
    for (int i = 0; i < network->neurons_hidden; i++) {
        double sum = 0.0;
        for (int j = 0; j < network->neurons_input; j++) {
            sum += inputs[j] * network->weights_hidden[j * network->neurons_hidden + i];
            // printf("input in forward pass is %f\n",inputs[j]);
            // printf("network weight in forward pass is %f\n",network->weights_hidden[j * network->neurons_hidden + i]);

        }
        // printf("sum is %f\n",sum);
        network->hiddenNeuron[i] = ReLU(sum + network->bias_hidden[i]);
        // printf("hidden network neuron is %f\n",network->hiddenNeuron[i]);
    }


    // for (int i = 0; i < network->neurons_output; i++) {
    //     printf("sum is %f for neuron %d\n", network->outputNeuron[i], i);
    // }

    // Forward pass from hidden to output layer
    for (int i = 0; i < network->neurons_output; i++) {
        double sum = 0.0; 
        for (int j = 0; j < network->neurons_hidden; j++) {
            sum += network->hiddenNeuron[j] * network->weights_output[j * network->neurons_output + i];
        }
        network->outputNeuron[i] = sum + network->bias_output[i];
    }


    softmax(network->outputNeuron, network->neurons_output);

    // for (int i = 0; i < network->neurons_output; i++) {
    //     printf("after softmax, number %d is %f\n", i, network->outputNeuron[i]);
    // }
}



void network_free(Network *network) {
    free(network->weights_hidden);
    free(network->weights_output);
    free(network->bias_output);
    free(network->bias_hidden);
    free(network->hiddenNeuron);
    free(network->outputNeuron);
}

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

void trainer_Mini_Batch_train(Trainer *trainer, Network *network, double **input, uint8_t *output, uint8_t epoch, uint32_t batch_size, double learning_rate, uint32_t dataset_size) {
    for (uint8_t e = 0; e < epoch; e++) {
        shuffle(input, output, dataset_size);
        uint32_t num_of_batches = (uint32_t) dataset_size / batch_size;

        double total_loss = 0.0;  // Initialize total loss for the epoch

        // printf("We're here yay\n");
        for (uint32_t n = 0; n < num_of_batches; n++) {
            double **inputBatch2D = get_batch_2D(input, batch_size, n);
            
            for (uint32_t b = 0; b < batch_size; b++) {
                double *inputBatch = flatten_2D(inputBatch2D, b);
                network_predict(network, inputBatch);


                double loss = cross_entropy_loss(network->outputNeuron, &output[b * network->neurons_output], network->neurons_output);
                total_loss += loss;

                backpropagation(network, &output[b * network->neurons_output], trainer, learning_rate, inputBatch);

                free(inputBatch);
            }

            // Free each image in the batch
            for (uint32_t b = 0; b < batch_size; b++) {
                free(inputBatch2D[b]);
            }

            // Free batch array
            free(inputBatch2D);
        }

        for (int i = 0; i < network->neurons_hidden; i++)
        {
            // printf("output neuron for neuron number %d is %.2f\n",i+1,network->outputNeuron[i]);
        }
        

        // Calculate and print average loss for the epoch
        double average_loss = total_loss / (num_of_batches * batch_size);
        printf("Epoch %d: Average Loss = %f\n", e + 1, average_loss);
    }
}





void trainer_free(Trainer* trainer){
    free(trainer->grad_hidden);
    free(trainer->grad_output);
}


