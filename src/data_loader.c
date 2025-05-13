#include "../include/data_loader.h"
#include <stdlib.h>

// convert high endian to low endian
uint32_t reverse_endian(uint32_t value) {
    return ((value & 0xFF) << 24) |
           ((value & 0xFF00) << 8) |
           ((value & 0xFF0000) >> 8) |
           ((value & 0xFF000000) >> 24);
}
double **load_data_file(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Read the header
    idx_header header;
    if (fread(&header.magic_number, sizeof(uint32_t), 1, file) != 1) {
        perror("Failed to read magic number");
        fclose(file);
        return NULL;
    }

    header.magic_number = reverse_endian(header.magic_number);

    if (header.magic_number != 2051) { // Magic number for image files
        fprintf(stderr, "Invalid magic number: %u. Expected 2051 for image file.\n", header.magic_number);
        fclose(file);
        return NULL;
    }

    // Read the number of dimensions
    uint32_t num_dimensions = 3; 
    uint32_t dimensions[num_dimensions];
    if (fread(dimensions, sizeof(uint32_t), num_dimensions, file) != num_dimensions) {
        perror("Failed to read dimensions");
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < num_dimensions; i++) {
        dimensions[i] = reverse_endian(dimensions[i]);
    }

    uint32_t num_images = dimensions[0];
    uint32_t rows = dimensions[1];
    uint32_t cols = dimensions[2];

    // Allocate memory for image data
    double **image_data = (double **)malloc(num_images * sizeof(double *));
    if (!image_data) {
        perror("Memory allocation failed for image data");
        fclose(file);
        return NULL;
    }

    for (uint32_t i = 0; i < num_images; i++) {
        image_data[i] = (double *)malloc(rows * cols * sizeof(double));
        if (!image_data[i]) {
            perror("Memory allocation failed for image row");
            for (uint32_t j = 0; j < i; j++) {
                free(image_data[j]);
            }
            free(image_data);
            fclose(file);
            return NULL;
        }
    }

    // Read the image data
    for (uint32_t i = 0; i < num_images; i++) {
        if (fread(image_data[i], sizeof(uint8_t), rows * cols, file) != rows * cols) {
            perror("Failed to read image data");
            for (uint32_t j = 0; j <= i; j++) {
                free(image_data[j]);
            }
            free(image_data);
            fclose(file);
            return NULL;
        }

        // Normalize pixel values to [0, 1]
        for (uint32_t j = 0; j < rows * cols; j++) {
            image_data[i][j] /= 255.0;
        }
    }

    fclose(file);
    return image_data;
}

uint8_t *load_text_file(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Couldn't open file");
        return NULL;
    }

    // Read the magic number
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) {
        perror("Failed to read magic number");
        fclose(file);
        return NULL;
    }

    magic_number = reverse_endian(magic_number);

    if (magic_number != 2049) { // Magic number for label files
        fprintf(stderr, "Invalid magic number: %u. Expected 2049 for label file.\n", magic_number);
        fclose(file);
        return NULL;
    }

    // Read the number of items
    uint32_t num_items;
    if (fread(&num_items, sizeof(uint32_t), 1, file) != 1) {
        perror("Failed to read number of items");
        fclose(file);
        return NULL;
    }

    num_items = reverse_endian(num_items);

    printf("Magic number: %u\n", magic_number);
    printf("Number of items: %u\n", num_items);

    // Allocate memory for label data
    uint8_t *data = (uint8_t *)malloc(num_items * sizeof(uint8_t));
    if (!data) {
        perror("Memory allocation failed for label data");
        fclose(file);
        return NULL;
    }

    // Read the label data
    if (fread(data, sizeof(uint8_t), num_items, file) != num_items) {
        perror("Failed to read label data");
        free(data);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return data;
}