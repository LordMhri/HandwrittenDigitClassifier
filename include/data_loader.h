#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <stdio.h>
#include <stdint.h>


typedef struct idx_header
{
    uint32_t magic_number;
    uint32_t num_items;
    uint32_t num_dims;
    uint32_t dim_sizes[3];
} idx_header ;

//load an idx file and return a 2D array of ubyte elements
uint8_t **load_data_file(const char *filename,uint32_t *num_items,uint32_t *num_dims,uint32_t *dim_sizes);

//convert high endian to low endian and vice versa
uint32_t reverse_endian(uint32_t value);

#endif