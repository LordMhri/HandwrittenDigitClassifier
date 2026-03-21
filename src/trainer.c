#include "../include/trainer.h"
#include "../include/utils.h"
#include "../include/network.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Trainer* trainer_init(Network *network, double learning_rate, int epochs, int batch_size,
                      int dataset_size, double **train_data, uint8_t *train_labels) {
    Trainer *trainer = (Trainer *)(malloc(sizeof(Trainer)));
    if (!trainer) {
        perror("malloc failed during trainer struct initialization");
        return NULL;
    }

    trainer->network = network;
    trainer->learning_rate = learning_rate;
    trainer->epochs = epochs;
    trainer->batch_size = batch_size;
    trainer->dataset_size = dataset_size;
    trainer->train_data = train_data;
    trainer->train_labels = train_labels;

    int num_transitions = network->num_layers - 1;
    trainer->acc_grad_w = calloc(num_transitions, sizeof(double *));
    trainer->acc_grad_b = calloc(num_transitions, sizeof(double *));
    trainer->deltas = calloc(num_transitions, sizeof(double *));

    if (!trainer->acc_grad_w || !trainer->acc_grad_b || !trainer->deltas) {
        perror("Error allocating trainer arrays");
        return NULL;
    }

    for (int i = 0; i < num_transitions; i++) {
        int input_size = network->layer_sizes[i];
        int output_size = network->layer_sizes[i + 1];

        trainer->acc_grad_w[i] = calloc(input_size * output_size, sizeof(double));
        trainer->acc_grad_b[i] = calloc(output_size, sizeof(double));
        trainer->deltas[i] = calloc(output_size, sizeof(double));
        
        if (!trainer->acc_grad_w[i] || !trainer->acc_grad_b[i] || !trainer->deltas[i]) {
            perror("Error allocating gradient/delta buffers");
            return NULL;
        }
    }

    return trainer;
}

void trainer_free(Trainer *trainer) {
    if (!trainer) return;
    int num_transitions = trainer->network->num_layers - 1;
    for (int i = 0; i < num_transitions; i++) {
        free(trainer->acc_grad_w[i]);
        free(trainer->acc_grad_b[i]);
        free(trainer->deltas[i]);
    }
    free(trainer->acc_grad_w);
    free(trainer->acc_grad_b);
    free(trainer->deltas);
    free(trainer);
}

double ReLU_Prime(double x) {
    return x > 0 ? 1 : 0;
}

void forward_propagation(Network *network, double *inputs) {
    for (int i = 0; i < network->num_layers - 1; i++) {
        double *source_layer = (i == 0) ? inputs : network->neurons[i - 1];
        double *target_layer = network->neurons[i];
        double *weight_matrix = network->weights[i];
        double *bias_vector = network->biases[i];

        int input_size = network->layer_sizes[i];
        int output_size = network->layer_sizes[i + 1];

        matrix_multiply(target_layer, source_layer, weight_matrix, 1, input_size, output_size);

        for (int j = 0; j < output_size; j++) {
            target_layer[j] += bias_vector[j];
            if (i < network->num_layers - 2) {
                target_layer[j] = ReLU(target_layer[j]);
            }
        }
    }

    int output_layer_idx = network->num_layers - 2;
    int output_size = network->layer_sizes[network->num_layers - 1];
    softmax(network->neurons[output_layer_idx], output_size);
}

void backpropagation(Trainer *trainer, uint8_t actual_label, double *inputs) {
    Network *network = trainer->network;
    int num_layers = network->num_layers;
    int output_layer_idx = num_layers - 2;

    // Output Layer Delta (Softmax + Cross-Entropy)
    int output_size = network->layer_sizes[num_layers - 1];
    for (int i = 0; i < output_size; i++) {
        double y = (i == actual_label) ? 1.0 : 0.0;
        trainer->deltas[output_layer_idx][i] = network->neurons[output_layer_idx][i] - y;
    }

    // Backpropagate deltas to hidden layers
    for (int i = output_layer_idx - 1; i >= 0; i--) {
        int current_layer_size = network->layer_sizes[i + 1];
        int next_layer_size = network->layer_sizes[i + 2];

        for (int j = 0; j < current_layer_size; j++) {
            double error = 0.0;
            for (int k = 0; k < next_layer_size; k++) {
                error += trainer->deltas[i + 1][k] * network->weights[i + 1][j * next_layer_size + k];
            }
            trainer->deltas[i][j] = error * ReLU_Prime(network->neurons[i][j]);
        }
    }

    // Accumulate Gradients
    for (int i = 0; i < num_layers - 1; i++) {
        int input_size = network->layer_sizes[i];
        int output_size = network->layer_sizes[i + 1];
        double *source_neurons = (i == 0) ? inputs : network->neurons[i - 1];

        for (int j = 0; j < input_size; j++) {
            for (int k = 0; k < output_size; k++) {
                trainer->acc_grad_w[i][j * output_size + k] += trainer->deltas[i][k] * source_neurons[j];
            }
        }

        for (int j = 0; j < output_size; j++) {
            trainer->acc_grad_b[i][j] += trainer->deltas[i][j];
        }
    }
}

void apply_gradients(Trainer *trainer, uint32_t batch_size) {
    Network *network = trainer->network;
    for (int i = 0; i < network->num_layers - 1; i++) {
        int input_size = network->layer_sizes[i];
        int output_size = network->layer_sizes[i + 1];

        for (int j = 0; j < input_size * output_size; j++) {
            network->weights[i][j] -= (trainer->learning_rate * trainer->acc_grad_w[i][j]) / batch_size;
            trainer->acc_grad_w[i][j] = 0;
        }

        for (int j = 0; j < output_size; j++) {
            network->biases[i][j] -= (trainer->learning_rate * trainer->acc_grad_b[i][j]) / batch_size;
            trainer->acc_grad_b[i][j] = 0;
        }
    }
}

void trainer_train(Trainer *trainer) {
    for (int e = 0; e < trainer->epochs; e++) {
        shuffle(trainer->train_data, trainer->train_labels, trainer->dataset_size);
        double total_loss = 0.0;

        for (int i = 0; i < trainer->dataset_size; i++) {
            forward_propagation(trainer->network, trainer->train_data[i]);
            total_loss += cross_entropy_loss(trainer->network->neurons[trainer->network->num_layers - 2], trainer->train_labels[i]);
            backpropagation(trainer, trainer->train_labels[i], trainer->train_data[i]);

            if ((i + 1) % trainer->batch_size == 0 || (i + 1) == trainer->dataset_size) {
                int current_batch_size = (i + 1) % trainer->batch_size == 0 ? trainer->batch_size : (i + 1) % trainer->batch_size;
                apply_gradients(trainer, current_batch_size);
            }
        }
        printf("Epoch %d/%d: Average Loss = %f\n", e + 1, trainer->epochs, total_loss / trainer->dataset_size);
    }
}
