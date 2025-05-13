#include "../include/network.h"
#include "../include/trainer.h" 
#include "../include/data_loader.h"
#include "../include/utils.h" 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define BATCH_SIZE 32 
#define LEARNING_RATE 0.0001 
#define EPOCHS 10 
#define DATASET_SIZE 15 

int main() {

    const char *inputTrainDataPath = "../dataset/train-images.idx3-ubyte";
    const char *inputLabelDataPath = "../dataset/train-labels.idx1-ubyte";


    // Load data
    double **inputTrainData = load_data_file(inputTrainDataPath);
    if (inputTrainData == NULL) {
        fprintf(stderr, "Failed to load training image data. Exiting.\n");
        return EXIT_FAILURE;
    }
    uint8_t *inputLabelData = load_text_file(inputLabelDataPath);
    if (inputLabelData == NULL) {
        fprintf(stderr, "Failed to load training label data. Exiting.\n");
        // Free image data if label loading fails to prevent memory leak
        for (int i = 0; i < DATASET_SIZE; i++) {
            free(inputTrainData[i]);
        }
        free(inputTrainData);
        return EXIT_FAILURE;
    }

    printf("Data loading successful.\n");

    // Initialize network struct
    Network network = {0};


    if (network_init(&network,28*28,512,128,10) != 0 )
    {
        perror("An error has occurred while initializing the network");
        perror("Aborting...");
        network_free(&network); 
        return EXIT_FAILURE;
    }

    printf("Network initialized successfully.\n");



    for (int img_idx = 0; img_idx < DATASET_SIZE; img_idx++) {

        network_predict(&network, inputTrainData[img_idx]);

        // Print the prediction (output layer values)
        printf("Prediction for image %d (output layer values):\n", img_idx);
        for (int j = 0; j < network.output_neurons_num; j++) {
            printf("%.4f ", network.output_neurons[j]);
        }
        printf("\n");

        // Print the actual correct number
        printf("Actual number: %d\n", inputLabelData[img_idx]);

        // this is purely because i love seeing the data come to life
        // REMINDER: should be commented out on production
        printf("Image data visualization:\n");
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                // Flattening a 2D matrix into a 1D array
                double pixel = inputTrainData[img_idx][row * 28 + col];
                if (pixel > 0.01) { 
                    printf("\e[1;34m%.2f\e[0m ", pixel); 
                } else {
                    printf("\e[1;31m%.2f\e[0m ", pixel);
                }
            }
            printf("\n");
        }
        printf("\n");
    }


    // Free the input data
    for (int i = 0; i < DATASET_SIZE; i++) {
        free(inputTrainData[i]);
    }
    free(inputTrainData);
    free(inputLabelData);

    // Free the network memory
    network_free(&network);



    return EXIT_SUCCESS;
}