#include "../include/data_loader.h"
#include <stdlib.h>
#include <stdio.h>

int main() {
    const char *filename = "../dataset/train-images.idx3-ubyte"; 
    uint32_t num_items, num_rows, num_cols;
    

    uint8_t **images = load_data_file(filename, &num_items, &num_rows, &num_cols);
    

    if (!images) {
        fprintf(stderr, "Failed to load data from %s\n", filename);
        return EXIT_FAILURE;
    }

    printf("Loaded %u images of size %u x %u\n", num_items, num_rows, num_cols);
    
    //free memory allocated to each image
    for (uint32_t i = 0; i < num_items; i++) {
        free(images[i]);
    }
    //free images array
    free(images);
    
    return EXIT_SUCCESS;
}

