#include "../include/data_loader.h"
#include <stdlib.h>

// convert high endian to low endian
uint32_t reverse_endian(uint32_t value) {
    return ((value & 0xFF) << 24) |
           ((value & 0xFF00) << 8) |
           ((value & 0xFF0000) >> 8) |
           ((value & 0xFF000000) >> 24);
};

uint8_t **load_data_file(const char *filename) {
    //opens file in read only binary mode hence "rb"
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Read magic number (4 bytes)
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(magic_number), 1, file) != 1) {
        perror("Failed to read magic number");
        fclose(file);
        return NULL;
    }
    magic_number = reverse_endian(magic_number);

    //magic number for idx files containing images is 2051
    if (magic_number != 2051) {
        perror("Invalid IDX Format magic number");
        fclose(file);
        return NULL;
    }

    //number of items aka image in the binary file
    uint32_t items;
    if (fread(&items, sizeof(items), 1, file) != 1) {
        perror("Failed to read number of items");
        fclose(file);
        return NULL;
    }
    items = reverse_endian(items);

    //number of rows(height) of each image 
    uint32_t rows;
    if (fread(&rows, sizeof(rows), 1, file) != 1) {
        perror("Failed to read number of rows");
        fclose(file);
        return NULL;
    }
    rows = reverse_endian(rows);

    //number of cols(width) of each image
    uint32_t cols;
    if (fread(&cols, sizeof(cols), 1, file) != 1) {
        perror("Failed to read number of columns");
        fclose(file);
        return NULL;
    }
    cols = reverse_endian(cols);

    

    printf("Magic number: %u\n", magic_number);
    printf("Number of items: %u\n", items);
    printf("Number of rows: %u\n", rows);
    printf("Number of columns: %u\n", cols);

    // Allocate memory for the image data (2D images: items x (rows * cols))
    uint8_t **data = (uint8_t **) malloc(items * sizeof(uint8_t *));


    for (uint32_t i = 0; i < items; i++) {
        //allocates memory for each item (image)
        //each image has height and width so allocates memory of size rows*cols
        data[i] = (uint8_t *)malloc(rows * cols * sizeof(uint8_t));
        if (fread(data[i], sizeof(uint8_t), rows * cols, file) != rows * cols) {
            perror("Failed to read image data");
            fclose(file);
            return NULL;
        }
    }

    fclose(file);
    return data;
}

uint8_t* load_text_file(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Couldn't open file");
        return NULL;
    }

    uint32_t magic_number;
    if (fread(&magic_number, sizeof(magic_number), 1, file) != 1) {
        perror("Failed to read magic number");
        fclose(file);
        return NULL;
    }
    magic_number = reverse_endian(magic_number);

    if (magic_number != 2049) {

        perror("Invalid IDX magic_number");
        fclose(file);
        return NULL;
    }

    uint32_t num_items;
    if (fread(&num_items, sizeof(num_items), 1, file) != 1) {
        perror("Failed to read number of items");
        fclose(file);
        return NULL;
    }
    num_items = reverse_endian(num_items);


    uint8_t *data = (uint8_t *)malloc(num_items * sizeof(uint8_t));
    if (fread(data, sizeof(uint8_t), num_items, file) != num_items) {
        perror("Failed to read label data");
        fclose(file);
        free(data);
        return NULL;
    }

    printf("Magic number: %u\n", magic_number);
    printf("Number of items: %u\n",num_items);
    fclose(file);
    return data;
}