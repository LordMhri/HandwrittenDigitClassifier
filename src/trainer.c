#include "../include/trainer.h"
#include "../include/utils.h"
#include "../include/network.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Trainer* trainer_init(Network *network,double learning_rate,int epochs,int batch_size,
                      int dataset_size,double **train_data,uint8_t* train_labels)
                      {

    Trainer *trainer = (Trainer* )(malloc(sizeof(Trainer)));
    if(!trainer) {
        perror("malloc failed during trainer struct initialization");
        return NULL;
    } else {
        printf("trainer malloc worked\n");
    }

    //assign the parameters to the given network
    trainer->network = network;
    trainer->learning_rate = learning_rate;
    trainer->epochs = epochs;
    trainer->batch_size = batch_size;
    trainer->dataset_size = dataset_size;
    trainer->train_data = train_data;
    trainer->train_labels = train_labels;


  trainer->acc_grad_b1 = calloc(network->first_hidden_neuron_num,sizeof(double));
  trainer->acc_grad_w1 = calloc((network->first_hidden_neuron_num * network->neurons_input),sizeof(double));

  trainer->acc_grad_b2 = calloc(network->second_hidden_neuron_num,sizeof(double));
  trainer->acc_grad_w2 = calloc((network->first_hidden_neuron_num * network->second_hidden_neuron_num),sizeof(double));

  trainer->acc_grad_b3 = calloc(network->output_neurons_num,sizeof(double));
  trainer->acc_grad_w3 = calloc((network->second_hidden_neuron_num * network->output_neurons_num),sizeof(double));

    return  trainer;
}




double ReLU_Prime(double x) {
    return x > 0 ? 1 : 0;
}


void forward_propagation(Network *network,double *inputs) {
    matrix_multiply(network->first_hidden_neurons,
                    inputs,
                    network->input_to_first_weight,
                    1,
                    network->neurons_input,
                    network->first_hidden_neuron_num);

    // Add biases to the first hidden layer and apply ReLU activation
    for (int i = 0; i < network->first_hidden_neuron_num; i++) {
        network->first_hidden_neurons[i] += network->input_to_first_bias[i];
        network->first_hidden_neurons[i] = ReLU(network->first_hidden_neurons[i]);
    }

    // From first hidden layer to second hidden layer
    matrix_multiply(network->second_hidden_neurons,
                    network->first_hidden_neurons,
                    network->first_to_second_weight,
                    1,
                    network->first_hidden_neuron_num,
                    network->second_hidden_neuron_num);

    // Add biases to the second hidden layer and apply ReLU activation
    for (int i = 0; i < network->second_hidden_neuron_num; i++) {
        network->second_hidden_neurons[i] += network->first_to_second_bias[i];
        network->second_hidden_neurons[i] = ReLU(network->second_hidden_neurons[i]);
    }

    // From second hidden layer to output layer
    matrix_multiply(network->output_neurons,
                    network->second_hidden_neurons,
                    network->second_to_output_weight,
                    1,
                    network->second_hidden_neuron_num,
                    network->output_neurons_num);

    // Add biases to the output layer
    for (int i = 0; i < network->output_neurons_num; i++) {
        network->output_neurons[i] += network->second_to_output_bias[i];
    }

    // Apply softmax activation to the output layer
    softmax(network->output_neurons, network->output_neurons_num);
}

void backpropagate() {

}

//
//
//void apply_gradients(Trainer *trainer, Network *network, double learning_rate, uint32_t batch_size) {
//    // Update weights for the output layer
//    for (int i = 0; i < network->neurons_hidden; i++)
//    {
//        for (int j = 0; j < network->neurons_output; j++)
//        {
//            network->weights_output[i * network->neurons_output + j] -= learning_rate * (trainer->acc_grad_weights_output[i * network->neurons_output + j] / batch_size); // Apply averaged gradient
//            trainer->acc_grad_weights_output[i * network->neurons_output + j] = 0.0; // Reset accumulator
//        }
//    }
//
//    // Update biases for the output layer
//    for (int i = 0; i < network->neurons_output; i++)
//    {
//        network->bias_output[i] -= learning_rate * (trainer->acc_grad_bias_output[i] / batch_size); // Apply averaged gradient
//        trainer->acc_grad_bias_output[i] = 0.0; // Reset accumulator
//    }
//
//    // Update weights for the hidden layer
//    for (int i = 0; i < network->neurons_input; i++)
//    {
//        for (int j = 0; j < network->neurons_hidden; j++)
//        {
//            network->weights_hidden[i * network->neurons_hidden + j] -= learning_rate * (trainer->acc_grad_weights_hidden[i * network->neurons_hidden + j] / batch_size); // Apply averaged gradient
//            trainer->acc_grad_weights_hidden[i * network->neurons_hidden + j] = 0.0; // Reset accumulator
//        }
//    }
//
//    // Update biases for the hidden layer
//    for (int i = 0; i < network->neurons_hidden; i++)
//    {
//        network->bias_hidden[i] -= learning_rate * (trainer->acc_grad_bias_hidden[i] / batch_size); // Apply averaged gradient
//        trainer->acc_grad_bias_hidden[i] = 0.0; // Reset accumulator
//    }
//}
//
//void trainer_Mini_Batch_train(Trainer *trainer, Network *network, double **input, uint8_t *output, uint8_t epoch, uint32_t batch_size, double learning_rate, uint32_t  dataset_size)
//{
//    for (uint8_t e = 0; e < epoch; e++)
//    {
//        shuffle(input, output, dataset_size);
//
//        uint32_t num_of_batches = (uint32_t)dataset_size / batch_size;
//        double total_loss = 0.0; // Initialize total loss for the epoch
//
//        for (uint32_t n = 0; n < num_of_batches; n++)
//        {
//            double **inputBatch2D = get_batch_2D(input, batch_size, n);
//
//            for (uint32_t b = 0; b < batch_size; b++)
//            {
//                double *inputBatch = flatten_2D(inputBatch2D, b);
//                network_predict(network, inputBatch);
//
//
//                double loss = cross_entropy_loss(network->outputNeuron, output[n * batch_size + b]);
//
//                total_loss += loss;
//
//                backpropagation(network, output[n * batch_size + b], trainer, inputBatch);
//
//                free(inputBatch);
//            }
//
//            // Apply gradients after processing the entire batch
//            apply_gradients(trainer, network, learning_rate, batch_size);
//
//
//            // Free each image in the batch
//            for (uint32_t b = 0; b < batch_size; b++)
//            {
//                free(inputBatch2D[b]);
//            }
//            free(inputBatch2D);
//        }
//
//        double average_loss = total_loss / num_of_batches;
//        printf("Epoch %d: Average Loss = %f\n", e + 1, average_loss);
//    }
//}
//
//
//void trainer_free(Trainer* trainer){
//    free(trainer->acc_grad_weights_hidden);
//    free(trainer->acc_grad_bias_hidden);
//    free(trainer->acc_grad_weights_output);
//    free(trainer->acc_grad_bias_output);
//}

