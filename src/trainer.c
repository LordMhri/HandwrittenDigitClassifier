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
    printf("  Entering forward_propagation\n");

    printf("  Starting Layer 1 forward pass\n");
    matrix_multiply(network->first_hidden_pre_activation_values,
                    inputs,
                    network->input_to_first_weight,
                    1,
                    network->neurons_input,
                    network->first_hidden_neuron_num);

    matrix_addition(network->first_hidden_pre_activation_values,
                    network->first_hidden_pre_activation_values,
                    network->input_to_first_bias,
                    1,
                    network->first_hidden_neuron_num
                    );

    // apply ReLU to each neuron
    for (int i = 0; i < network->first_hidden_neuron_num; i++) {
        network->first_hidden_neurons[i] = ReLU(network->first_hidden_pre_activation_values[i]);
    }
    printf("  Finished Layer 1 forward pass\n");

    printf("  Starting Layer 2 forward pass\n");
    // From first hidden layer to second hidden layer
    matrix_multiply(network->second_hidden_pre_activation_values,
                    network->first_hidden_neurons,
                    network->first_to_second_weight,
                    1,
                    network->first_hidden_neuron_num,
                    network->second_hidden_neuron_num);

    matrix_addition(network->second_hidden_pre_activation_values,
                    network->first_hidden_neurons,
                    network->first_to_second_bias,
                    1,
                    network->second_hidden_neuron_num
    );


    // Add biases to the second hidden layer and apply ReLU activation
    for (int i = 0; i < network->second_hidden_neuron_num; i++) {
        network->second_hidden_neurons[i] = ReLU(network->second_hidden_pre_activation_values[i]);
    }
    printf("  Finished Layer 2 forward pass\n");

    printf("  Starting Output Layer forward pass\n");
    // From second hidden layer to output layer
    matrix_multiply(network->output_pre_activation_values,
                    network->second_hidden_neurons,
                    network->second_to_output_weight,
                    1,
                    network->second_hidden_neuron_num,
                    network->output_neurons_num);

    matrix_addition(network->output_pre_activation_values,
                    network->output_pre_activation_values,
                    network->second_to_output_bias,
                    1,
                    network->output_neurons_num
    );

    // Add biases to the output layer
    for (int i = 0; i < network->output_neurons_num; i++) {
        network->output_neurons[i] = ReLU(network->output_pre_activation_values[i]);

    }
    printf("  Finished Output Layer forward pass\n");

    // Apply softmax activation to the output layer
    softmax(network->output_pre_activation_values,network->output_neurons, network->output_neurons_num);
    printf("  Exiting forward_propagation\n");
}

void backpropagate(Trainer  *trainer,Network *network,double *inputs,uint8_t label) {


    printf("Starting backprop");
    //TODO: calloc checks all over this function
    //calculate error caused by z3 which is the last layer
    //basically trying to find the derivative of the cost function with respect to z3
    double *delta3 = calloc(network->output_neurons_num,sizeof(double ));
    printf("Allocating memory for delta3");

    //uint8_t chosen because the values are guaranteed to be low
    uint8_t* one_hot_y = calloc(network->output_neurons_num,sizeof(uint8_t));
    one_hot_encode(label,one_hot_y);

    //error caused by z3 is the difference between the predicted values
    //and the one_hot encoded value
    for (int i = 0; i < network->output_neurons_num; ++i) {
        delta3[i] = network->output_neurons[i] -one_hot_y[i];
    }

    free(one_hot_y); //one_hot_y is no longer needed

    for (int i = 0; i < network->output_neurons_num; ++i) {
        for (int j = 0; j < network->second_hidden_neuron_num; ++j) {
            trainer->acc_grad_w3[i*network->second_hidden_neuron_num + j] += delta3[i] * network->second_hidden_neurons[j];
        }
    }

    matrix_addition(trainer->acc_grad_w3,trainer->acc_grad_w3,delta3,network->output_neurons_num,1);

    //weightin3^T * delta3
    double  *prop_error_2 = calloc(network->second_hidden_neuron_num,sizeof(double));
    printf("Allocating memory for prop_error_2");

    matrix_transpose_vector_multiply(prop_error_2,network->second_to_output_weight,delta3,network->output_neurons_num,network->second_hidden_neuron_num);


    double *delta2 = calloc(network->second_hidden_neuron_num,sizeof(double));
    printf("Allocating memory for delta2");

    for (int i = 0; i < network->second_hidden_neuron_num; i++) {
        double relu_prime = ReLU_Prime(network->second_hidden_pre_activation_values[i]);
        delta2[i] = prop_error_2[i] * relu_prime;
    }
    free(prop_error_2);

    for (int i = 0; i < network->second_hidden_neuron_num; ++i) {
        for (int j = 0; j < network->first_hidden_neuron_num; ++j) {
            trainer->acc_grad_w2[i*network->first_hidden_neuron_num + j] = delta2[i] * network->first_hidden_neurons[j];
        }
    }

    matrix_addition(trainer->acc_grad_w2,trainer->acc_grad_w3,delta2,network->second_hidden_neuron_num,1);

    double *prop_error_1 = calloc(network->first_hidden_neuron_num, sizeof(double ));
    printf("Allocating memory for prop_error_1");


    matrix_transpose_vector_multiply(prop_error_1,network->first_to_second_weight,delta2,network->second_hidden_neuron_num,network->first_hidden_neuron_num);

    double *delta1 = calloc(network->first_hidden_neuron_num, sizeof(double));
    printf("Allocating memory for delta1");


    for (int i = 0; i < network->first_hidden_neuron_num; i++) {
        double relu_prime = ReLU_Prime(network->first_hidden_pre_activation_values[i]);
        delta1[i] = prop_error_1[i] * relu_prime;
    }
    free(prop_error_1);



    for (int i = 0; i < network->first_hidden_neuron_num; i++) {
        for (int j = 0; j < network->neurons_input; j++) {
            trainer->acc_grad_w1[i * network->neurons_input + j] += delta1[i] * inputs[j];
        }
    }


    matrix_addition(trainer->acc_grad_b1,
                    trainer->acc_grad_b1,
                    delta1,
                    network->first_hidden_neuron_num,
                    1);


    free(delta1);

    free(delta3);
    free(delta2);

}



void apply_gradients(Trainer *trainer, Network *network, double learning_rate, uint32_t batch_size) {
    // Calculate the scaling factor (learning_rate / batch_size)
    double scale = learning_rate / batch_size;

    // --- Update Weights and Biases for Input -> First Hidden Layer (Layer 1) ---

    // Update Weights (W1)
    // Dimensions of W1 and acc_grad_w1: first_hidden_neuron_num x neurons_input
    for (int i = 0; i < network->first_hidden_neuron_num; i++) { // Iterate through rows
        for (int j = 0; j < network->neurons_input; j++) { // Iterate through columns
            int index = i * network->neurons_input + j; // Index in the flattened array
            // Apply the update rule: W_new = W_old - scale * acc_grad_W
            network->input_to_first_weight[index] -= scale * trainer->acc_grad_w1[index];
            // Reset the accumulated gradient for this parameter
            trainer->acc_grad_w1[index] = 0.0;
        }
    }

    // Update Biases (b1)
    // Dimensions of b1 and acc_grad_b1: first_hidden_neuron_num
    for (int i = 0; i < network->first_hidden_neuron_num; i++) { // Iterate through elements
        // Apply the update rule: b_new = b_old - scale * acc_grad_b
        network->input_to_first_bias[i] -= scale * trainer->acc_grad_b1[i];
        // Reset the accumulated gradient for this parameter
        trainer->acc_grad_b1[i] = 0.0;
    }


    // --- Update Weights and Biases for First Hidden -> Second Hidden Layer (Layer 2) ---

    // Update Weights (W2)
    // Dimensions of W2 and acc_grad_w2: second_hidden_neuron_num x first_hidden_neuron_num
    for (int i = 0; i < network->second_hidden_neuron_num; i++) { // Iterate through rows
        for (int j = 0; j < network->first_hidden_neuron_num; j++) { // Iterate through columns
            int index = i * network->first_hidden_neuron_num + j; // Index in the flattened array
            // Apply the update rule: W_new = W_old - scale * acc_grad_W
            network->first_to_second_weight[index] -= scale * trainer->acc_grad_w2[index];
            // Reset the accumulated gradient for this parameter
            trainer->acc_grad_w2[index] = 0.0;
        }
    }

    // Update Biases (b2)
    // Dimensions of b2 and acc_grad_b2: second_hidden_neuron_num
    for (int i = 0; i < network->second_hidden_neuron_num; i++) { // Iterate through elements
        // Apply the update rule: b_new = b_old - scale * acc_grad_b
        network->first_to_second_bias[i] -= scale * trainer->acc_grad_b2[i];
        // Reset the accumulated gradient for this parameter
        trainer->acc_grad_b2[i] = 0.0;
    }


    // --- Update Weights and Biases for Second Hidden -> Output Layer (Layer 3) ---

    // Update Weights (W3)
    // Dimensions of W3 and acc_grad_w3: output_neurons_num x second_hidden_neuron_num
    for (int i = 0; i < network->output_neurons_num; i++) { // Iterate through rows
        for (int j = 0; j < network->second_hidden_neuron_num; j++) { // Iterate through columns
            int index = i * network->second_hidden_neuron_num + j; // Index in the flattened array
            // Apply the update rule: W_new = W_old - scale * acc_grad_W
            network->second_to_output_weight[index] -= scale * trainer->acc_grad_w3[index];
            // Reset the accumulated gradient for this parameter
            trainer->acc_grad_w3[index] = 0.0;
        }
    }

    // Update Biases (b3)
    // Dimensions of b3 and acc_grad_b3: output_neurons_num
    for (int i = 0; i < network->output_neurons_num; i++) { // Iterate through elements
        // Apply the update rule: b_new = b_old - scale * acc_grad_b
        network->second_to_output_bias[i] -= scale * trainer->acc_grad_b3[i];
        // Reset the accumulated gradient for this parameter
        trainer->acc_grad_b3[i] = 0.0;
    }
}
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

void trainer_free(Trainer* trainer){
    free(trainer->acc_grad_w3);
    free(trainer->acc_grad_w2);
    free(trainer->acc_grad_w1);
    free(trainer->acc_grad_b3);
    free(trainer->acc_grad_b2);
    free(trainer->acc_grad_b1);

}

