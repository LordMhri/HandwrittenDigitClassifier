#include "../include/utils.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define M_PI 3.14159265358979323846


void shuffle_data(double **data, uint8_t *labels, int n_samples) {
    srand(time(NULL)); // Seed random number generator

    for (int i = n_samples - 1; i > 0; i--) {
        // Generate random index between 0 and i
        int j = rand() % (i + 1);


        double *temp_data = data[i];
        data[i] = data[j];
        data[j] = temp_data;

        uint8_t temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;
    }
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

void clip_gradients(double *grad, int size, double threshold) {
    for (int i = 0; i < size; i++) {
        if (grad[i] > threshold) grad[i] = threshold;
        if (grad[i] < -threshold) grad[i] = -threshold;
    }
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

void matrix_transpose_vector_multiply(double *result, double *matrix,
                                      double *vector, int num_rows, int num_cols) {
    for (int col = 0; col < num_cols; col++) {
        result[col] = 0.0;
        for (int row = 0; row < num_rows; row++) {
            result[col] += matrix[row * num_cols + col] * vector[row];
        }
    }
}


void one_hot_encode(uint8_t label, uint8_t *array) {
    memset(array, 0, 10 * sizeof(uint8_t));
    array[label] = 1;
}

//he_intialization of random weights
void he_init(double *weights, int neurons_output, int neurons_input) {
    double stddev = sqrt(2.0 / neurons_input);
    for (int i = 0; i < neurons_input * neurons_output; i++) {
        //new box-muller transform for gauss distribution
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

        weights[i] = z * stddev; // Proper zero-mean Gaussian
    }
}


void softmax(double *input_z, double *output, int length) {

    if (length <= 0) return;
    double max_input = input_z[0];
    for (int i = 1; i < length; i++) {
        if (input_z[i] > max_input) {
            max_input = input_z[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input_z[i] - max_input);
        sum += output[i];
    }

    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

double ReLU(double x) {
    return (x > 0) ? x : 0.01 * x;  // Leaky ReLU
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
