#include "../include/utils.h"
#include "../include/network.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void save_network_to_json(struct Network *network, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file for saving network");
        return;
    }

    fprintf(file, "{\n");
    fprintf(file, "  \"num_layers\": %d,\n", network->num_layers);
    
    // Layer sizes
    fprintf(file, "  \"layer_sizes\": [");
    for (int i = 0; i < network->num_layers; i++) {
        fprintf(file, "%d%s", network->layer_sizes[i], (i == network->num_layers - 1) ? "" : ", ");
    }
    fprintf(file, "],\n");

    // Weights
    fprintf(file, "  \"weights\": [\n");
    for (int i = 0; i < network->num_layers - 1; i++) {
        fprintf(file, "    [");
        int size = network->layer_sizes[i] * network->layer_sizes[i+1];
        for (int j = 0; j < size; j++) {
            fprintf(file, "%.10f%s", network->weights[i][j], (j == size - 1) ? "" : ", ");
        }
        fprintf(file, "]%s\n", (i == network->num_layers - 2) ? "" : ",");
    }
    fprintf(file, "  ],\n");

    // Biases
    fprintf(file, "  \"biases\": [\n");
    for (int i = 0; i < network->num_layers - 1; i++) {
        fprintf(file, "    [");
        int size = network->layer_sizes[i+1];
        for (int j = 0; j < size; j++) {
            fprintf(file, "%.10f%s", network->biases[i][j], (j == size - 1) ? "" : ", ");
        }
        fprintf(file, "]%s\n", (i == network->num_layers - 2) ? "" : ",");
    }
    fprintf(file, "  ]\n");

    fprintf(file, "}\n");
    fclose(file);
    printf("Network saved to %s\n", filename);
}


double** normalize_image_data(uint8_t **inputs, int number_of_images) {

    double** normalized_data = (double**) calloc(number_of_images, sizeof(double*));

    for (int img_idx = 0; img_idx < number_of_images; img_idx++) {
        normalized_data[img_idx] = (double*) malloc(28 * 28 * sizeof(double));

        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                normalized_data[img_idx][row * 28 + col] = (double) inputs[img_idx][row * 28 + col] / 255.0;
            }
        }
    }

    return normalized_data; 
}
void matrix_multiply(double *result, double *first_matrix, double *second_matrix, int m, int n, int p) {
    for (int i = 0; i < m * p; i++) {
        result[i] = 0.0;
    }

    
    for (int i = 0; i < m; i++) { 
        for (int j = 0; j < p; j++) { 
            for (int k = 0; k < n; k++) { 
                result[i * p + j] += first_matrix[i * n + k] * second_matrix[k * p + j];
            }
        }
    }
}

void matrix_addition(double *result, double *matrix_a, double *matrix_b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * cols + j] = matrix_a[i * cols + j] + matrix_b[i * cols + j];
        }
    }
}

// Box-Muller transform: generates a standard normal random number
static double rand_normal() {
    double u1 = ((double) rand() + 1.0) / ((double) RAND_MAX + 1.0); // (0, 1]
    double u2 = ((double) rand() + 1.0) / ((double) RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// He initialization with proper Gaussian distribution: N(0, sqrt(2/fan_in))
void he_init(double *weights, int neurons_output, int neurons_input) {
    double stddev = sqrt(2.0 / neurons_input);
    for (int i = 0; i < neurons_input * neurons_output; i++) {
        weights[i] = rand_normal() * stddev;
    }
}


void softmax(double *input, int length) {

    if (length <= 0) return;
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

    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}   

double ReLU(double x) {
    return x > 0 ? x : 0;
}

double** get_batch_2D(double **input, int batch_size, int batch_index) {
    // Allocate memory for the batch of images
    double **batch = calloc(batch_size, sizeof(double*));
    // printf("We're called here!");

    for (int b = 0; b < batch_size; b++) {
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

void swap_double_ptrs(double** a,double** b){
    double *temp = *a;
    *a = *b;
    *b = temp;
}

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

double cross_entropy_loss(double *predicted, uint8_t actual_label_index) {
    return -log(fmax(predicted[actual_label_index], 1e-9)); 
}

double *flatten_2D(double **input, int index) {
    double *flattened = malloc(28 * 28 * sizeof(double));

    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            flattened[row * 28 + col] = input[index][row * 28 + col];
        }
    }

    return flattened;
}
