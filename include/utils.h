#ifndef UTILS_H
#define UTILS_H
#include <stdint.h>

double** normalize_image_data(uint8_t **inputs,int number_of_images);
void shuffle(uint8_t **inputs, uint8_t *output, int dataset_size);
double cross_entropy_loss(double *predicted, uint8_t *actual, int num_output);
double** get_batch_2D(uint8_t **input, uint32_t batch_size, uint8_t batch_index);

#endif