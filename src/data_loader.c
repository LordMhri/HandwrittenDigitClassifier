#include "../include/data_loader.h"
#include <stdlib.h>
#include <string.h>

// convert high endian to low endian
uint32_t reverse_endian(uint32_t value) {
    return ((value & 0xFF) << 24) |
           ((value & 0xFF00) << 8) |
           ((value & 0xFF0000) >> 8) |
           ((value & 0xFF000000) >> 24);
}

double **load_image_file(const char *filename, uint32_t *num_items, uint32_t *num_rows, uint32_t *num_cols) {
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

    
    *num_items = items;
    *num_rows = rows;
    *num_cols = cols;


    // Allocate memory for the image data (2D images: items x (rows * cols))
    uint8_t **data = (uint8_t **) malloc(items * sizeof(uint8_t *));

    //Allocate memory for the normalized data
    double **normalized_data = (double **) malloc(items * sizeof(double *));

    for (uint32_t i = 0; i < items; i++) {
        //allocates memory for each item (image)
        //each image has height and width so allocates memory of size rows*cols
        data[i] = (uint8_t *)malloc(rows * cols * sizeof(uint8_t));
        normalized_data[i] = (double *) malloc(rows * cols * sizeof(double));
        if (fread(data[i], sizeof(uint8_t), rows * cols, file) != rows * cols) {
            perror("Failed to read image data");
            fclose(file);
            return NULL;
        }

        //normalize data
        for (uint32_t j = 0; j <  rows*cols; j++)
        {
            //data values are between 0 and 255
            normalized_data[i][j] = data[i][j] / 255.0;
        }
        
        free(data[i]);
    }
    free(data);
    fclose(file);
    return normalized_data;
}


uint8_t* load_text_file(const char *filename,uint32_t *num_labels) {
        FILE *file = fopen(filename,"rb");
        if (!file)
        {
            perror("Failed to open file");
            return NULL;
        }

        uint32_t magic_number = 0;
        fread(&magic_number, sizeof(uint32_t), 1, file);
        magic_number = reverse_endian(magic_number); 
        if (magic_number != 2049)
        {
            perror("Invalid IDX magic number");
            return NULL;
        }

        
        fread(num_labels,sizeof(uint32_t),1,file);
        *num_labels = reverse_endian(*num_labels);

        uint8_t *labels = (uint8_t *)malloc(*num_labels * sizeof(uint8_t));
        if (fread(labels, sizeof(uint8_t), *num_labels, file) != *num_labels) {
            perror("Failed to read labels");
            free(labels);
            fclose(file);
            return NULL;
        }

    fclose(file);
    return labels;
        
}


void one_hot_encode(uint8_t label, double *target) {
    memset(target, 0, 10 * sizeof(double));  // Zero out the target vector
    target[label] = 1.0;  // Set the correct class index to 1
}