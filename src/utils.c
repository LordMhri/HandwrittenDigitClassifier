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

double cross_entropy_loss(double *predicted, uint8_t *actual, int num_output) {
    double loss = 0.0;
    for (int i = 0; i < num_output; i++) {
        loss -= actual[i] * log(fmax(predicted[actual[i]], 1e-9));
    }
    return loss / num_output;
}
