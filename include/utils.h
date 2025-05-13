#ifndef UTILS_H
#define UTILS_H
#include <stdint.h>


void softmax(double *input, int length);

void clip_gradients(double *gradients, int size, double threshold);

void he_init(double *weights,int neurons_output,int neurons_input);

double** normalize_image_data(uint8_t **inputs,int number_of_images);

void matrix_multiply(double *result, double *first_matrix,double *second_matrix,int m,int n,int p);

void shuffle(double **inputs, uint8_t *output, int dataset_size);

double cross_entropy_loss(double *predicted, uint8_t actual_label_index);

double** get_batch_2D(double **input, int batch_size, int batch_index);

double *flatten_2D(double **input, int index);

void swap_double_ptrs(double** a,double** b);

double ReLU(double x);

double ReLU_Prime(double x);



#endif