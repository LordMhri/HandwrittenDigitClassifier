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

    idx_header header; // Use the idx_header struct
    if (fread(&header.magic_number, sizeof(uint32_t), 1, file) != 1 || // Read header components directly into struct
        fread(&header.num_items, sizeof(uint32_t), 1, file) != 1 ||
        fread(&header.num_rows, sizeof(uint32_t), 1, file) != 1 || // Read rows
        fread(&header.num_cols, sizeof(uint32_t), 1, file) != 1)   // Read cols
    {
        perror("Failed to read IDX header");
        fclose(file);
        return NULL;
    }

    header.magic_number = reverse_endian(header.magic_number);
    header.num_items = reverse_endian(header.num_items);
    header.num_rows = reverse_endian(header.num_rows);
    header.num_cols = reverse_endian(header.num_cols);

    if (header.magic_number != 2051) { // Magic number for images
        fprintf(stderr, "Invalid magic number: %u. Expected 2051 for image file.\n", header.magic_number); // Use stderr for errors
        fclose(file);
        return NULL;
    }

    printf("Magic number: %u\n", header.magic_number);
    printf("Number of items: %u\n", header.num_items);
    printf("Number of rows: %u\n", header.num_rows);
    printf("Number of columns: %u\n", header.num_cols);

    double **data = (double **)malloc(header.num_items * sizeof(double *));
    if (!data) { // Check malloc success
        perror("Memory allocation failed for image data pointers");
        fclose(file);
        return NULL;
    }

    for (uint32_t i = 0; i < header.num_items; i++) {
        data[i] = (double *)malloc(header.num_rows * header.num_cols * sizeof(double));
        if (!data[i]) { // Check malloc success
            perror("Memory allocation failed for image data");
            // Clean up previously allocated memory
            for (uint32_t j = 0; j < i; j++) {
                free(data[j]);
            }
            free(data);
            fclose(file);
            return NULL;
        }
        uint8_t pixel_value; // Read as uint8_t first
        for (uint32_t j = 0; j < header.num_rows * header.num_cols; j++) {
            if (fread(&pixel_value, sizeof(uint8_t), 1, file) != 1) { // Read each pixel as uint8_t
                perror("Failed to read pixel data");
                // Clean up allocated memory
                for (uint32_t k = 0; k <= i; k++) {
                    free(data[k]);
                }
                free(data);
                fclose(file);
                return NULL;
            }
            data[i][j] = (double)pixel_value / 255.0; // Normalize to double here
        }
    }

    fclose(file);
    return data;
}

uint8_t *load_text_file(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Couldn't open file");
        return NULL;
    }

    idx_header header; // Use idx_header struct
    if (fread(&header.magic_number, sizeof(uint32_t), 1, file) != 1 || // Read header components directly into struct
        fread(&header.num_items, sizeof(uint32_t), 1, file) != 1)
    {
        perror("Failed to read IDX header");
        fclose(file);
        return NULL;
    }


    header.magic_number = reverse_endian(header.magic_number);
    header.num_items = reverse_endian(header.num_items);


    if (header.magic_number != 2049) { // Magic number for labels
        fprintf(stderr, "Invalid magic number: %u. Expected 2049 for label file.\n", header.magic_number); // Use stderr for errors
        fclose(file);
        return NULL;
    }

    printf("Magic number: %u\n", header.magic_number);
    printf("Number of items: %u\n", header.num_items);

    uint8_t *data = (uint8_t *)malloc(header.num_items * sizeof(uint8_t));
    if (!data) { // Check malloc success
        perror("Memory allocation failed for label data");
        fclose(file);
        return NULL;
    }

    if (fread(data, sizeof(uint8_t), header.num_items, file) != header.num_items) {
        perror("Failed to read label data");
        free(data);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return data;
}