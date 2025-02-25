#include "../include/utils.h"
#include <stdlib.h>
#include <math.h>

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

//he_intialization of random weights
void he_init(double *weights,int neurons_output,int neurons_input) {
    double stddev = sqrt(2.0/neurons_input);
    for (int i= 0; i < neurons_input*neurons_output; i++)
    {
        weights[i] = (double) rand() / RAND_MAX * stddev * 2 - stddev;
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
