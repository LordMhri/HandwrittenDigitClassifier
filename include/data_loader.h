#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <stdio.h>
#include <stdint.h>


typedef struct idx_header
{
    uint32_t magic_number;
    uint32_t dimensions[];
} idx_header ;

// Loads an IDX image file into memory
double **load_data_file(const char *filename);

// Converts a 32-bit unsigned integer from big-endian to little-endian byte order
// (or vice versa, depending on system architecture).
uint32_t reverse_endian(uint32_t value);

// Loads an IDX label file and returns a 1D array of unsigned byte elements (labels).
uint8_t *load_text_file(const char *filename);

#endif