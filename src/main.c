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
#define DATASET_SIZE 60000

int main() {

    const char *inputTrainDataPath = "../dataset/train-images.idx3-ubyte";
    const char *inputLabelDataPath = "../dataset/train-labels.idx1-ubyte";


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

    
    // srand(time(NULL));
    // shuffle((uint8_t **)inputTrainData, inputLabelData, DATASET_SIZE);
    
    // Network Initialization
    Network network = {0};
    if (network_init(&network, 28 * 28, 128, 10) != 0) { 
            fprintf(stderr, "Network initialization failed. Exiting.\n");
            // Free data before exiting
            for (int i = 0; i < DATASET_SIZE; i++) {
                    free(inputTrainData[i]);
                }
                free(inputTrainData);
                free(inputLabelData);
                return EXIT_FAILURE;
    }
    printf("Network initialized.\n");
    
    // Trainer Initialization
    Trainer trainer = {0};
    if (trainer_init(&trainer, &network) == NULL) {
            fprintf(stderr, "Trainer initialization failed. Exiting.\n");
            // Free network and data before exiting
            network_free(&network);
            for (int i = 0; i < DATASET_SIZE; i++) {
                    free(inputTrainData[i]);
                }
        free(inputTrainData);
        free(inputLabelData);
        return EXIT_FAILURE;
    }
    printf("Trainer initialized.\n");
    
    // Model Training
    printf("Starting training...\n");
    trainer_Mini_Batch_train(&trainer, &network, inputTrainData, inputLabelData, EPOCHS, BATCH_SIZE, LEARNING_RATE, DATASET_SIZE);
    printf("Training complete.\n");
    
    
    // Free resources
    printf("Freeing resources...\n");
    network_free(&network);
    trainer_free(&trainer);
    
    // Free input data
    for (int i = 0; i < DATASET_SIZE; i++) {
            free(inputTrainData[i]);
        }
        free(inputTrainData);
        free(inputLabelData);
        printf("Resources freed.\n");
        
        printf("Program finished.\n");
        return EXIT_SUCCESS;
    }
    
    
    // int num_to_load = dataset_size;
    // for (int img_idx = 0; img_idx < num_to_load; img_idx++) {
        //     printf("Number is %d\n", inputLabelData[img_idx]);
    //     // Loop through the rows and columns of each image
    //     for (int row = 0; row < 28; row++) {
    //         for (int col = 0; col < 28; col++) {
        //             // Flattening a 2D matrix into a 1D array
        //             double pixel = normalized_data[img_idx][row * 28 + col];
        //             if (pixel > 0) {
            //                 printf("\e[1;31m%.2f\e[0m ", pixel);
    //             } else {
        //                 printf("%.2f ", pixel);
        //             }
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }