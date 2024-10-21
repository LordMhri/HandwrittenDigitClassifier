#include "../include/utils.h"
#include <stdlib.h>

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
