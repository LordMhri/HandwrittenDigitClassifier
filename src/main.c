#include "../include/data_loader.h"
#include "../include/network.h"
#include <stdlib.h>
#include <stdio.h>

#define ITERS 1000
#define LEARNING_RATE 0.1

double compute_error(double *output, double *target, int size) {
    double error = 0.0;
    for (int i = 0; i < size; i++) {
        error += (output[i] - target[i]) * (output[i] - target[i]);
    }
    return error / size; 
}

int main() {
    const char *inputFileName = "../dataset/train-images.idx3-ubyte"; 
    const char *outputFileName = "../dataset/train-labels.idx1-ubyte";
    uint32_t num_items, num_rows, num_cols;
    
    // Load image data
    double **images = load_image_file(inputFileName, &num_items, &num_rows, &num_cols);
    if (!images) {
        perror("Failed to load file\n");
        return EXIT_FAILURE;
    }

    // Load label data
    uint32_t num_labels = 0;
    uint8_t *labels = load_text_file(outputFileName, &num_labels);

    // Initialize the network
    Network network = {0};
    network_init(&network, num_rows * num_cols, 5, 10);  // Input layer: 784, Hidden layer: 5, Output layer: 10
    Trainer trainer = {0};
    trainer_init(&trainer, &network);

    double target[10];  // One-hot encoded target

    // Training loop
    for (size_t iter = 0; iter < ITERS; iter++) {
        double total_error = 0.0;  // Accumulate total error for all images

        for (size_t img_idx = 0; img_idx < num_items; img_idx++) {
            // Flatten the image (28x28) to a 1D array (784 elements)
            double *input = images[img_idx];

            // One-hot encode the label
            one_hot_encode(labels[img_idx], target);

            // Train the network
            trainer_train(&trainer, &network, input, target, LEARNING_RATE);

            // Compute the error for this image
            total_error += compute_error(network.outputNeuron, target, 10);  // Using MSE for output error
        }

        // Print the average error for this iteration
        printf("Iteration %zu: Average Error = %f\n", iter, total_error / num_items);
    }

    // Free resources
    trainer_free(&trainer);
    network_free(&network);
    free(labels);
    
    for (uint32_t i = 0; i < num_items; i++) {
        free(images[i]);
    }
    free(images);
    
    return EXIT_SUCCESS;
}


