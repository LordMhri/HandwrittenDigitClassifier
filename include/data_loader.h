#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <stdio.h>
#include <stdint.h>


typedef struct idx_info
{
    uint32_t magic_number;
    uint32_t num_items;
    uint32_t rows;
    uint32_t columns;
} idx_info ;

// Loads an IDX image file and returns a 2D array of doubles (normalized 0.0-1.0)
double **load_image_file(const char *filename);

// Converts a 32-bit unsigned integer from big-endian to little-endian byte order
uint32_t reverse_endian(uint32_t value);

// Loads an IDX label file and returns a 1D array of labels (0-9)
uint8_t *load_label_file(const char *filename);

#endif
