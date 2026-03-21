#include "../include/network.h"
#include "../include/trainer.h" 
#include "../include/data_loader.h"
#include "../include/utils.h" 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define BATCH_SIZE 32 
#define LEARNING_RATE 0.001 
#define EPOCHS 25 
#define DATASET_SIZE 60000 
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

int main() {
    // Seed for repeatability during initialization
    srand(time(NULL));

    const char *inputTrainDataPath = "../dataset/train-images.idx3-ubyte";
    const char *inputLabelDataPath = "../dataset/train-labels.idx1-ubyte";

    printf("Loading full dataset (%d images)...\n", DATASET_SIZE);

    // Load full dataset
    double **inputTrainData = load_image_file(inputTrainDataPath);
    if (inputTrainData == NULL) {
        fprintf(stderr, "Failed to load training image data. Exiting.\n");
        return EXIT_FAILURE;
    }
    uint8_t *inputLabelData = load_label_file(inputLabelDataPath);
    if (inputLabelData == NULL) {
        fprintf(stderr, "Failed to load training label data. Exiting.\n");
        // Memory cleanup
        for (int i = 0; i < DATASET_SIZE; i++) {
            free(inputTrainData[i]);
        }
        free(inputTrainData);
        return EXIT_FAILURE;
    }

    printf("Dataset loaded successfully.\n");

    // Define the network architecture
    // 784 inputs -> 256 hidden -> 128 hidden -> 10 output
    int layer_sizes[] = {INPUT_SIZE, 256, 128, OUTPUT_SIZE};
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

    Network network = {0};
    if (network_init(&network, num_layers, layer_sizes) != 0 ) {
        fprintf(stderr, "An error has occurred while initializing the network. Aborting...\n");
        return EXIT_FAILURE;
    }

    printf("Network initialized with %d layers.\n", num_layers);

    Trainer *trainer = trainer_init(&network, LEARNING_RATE, EPOCHS,
                 BATCH_SIZE, DATASET_SIZE,
                 inputTrainData, inputLabelData);

    if (!trainer) {
        fprintf(stderr, "Failed to initialize trainer. Aborting...\n");
        return EXIT_FAILURE;
    }

    printf("Starting training pipeline...\n");
    clock_t start = clock();
    trainer_train(trainer);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Training complete in %.2f seconds.\n", time_spent);

    // Save model to JSON for web usage
    printf("Exporting model to JSON...\n");
    save_network_to_json(&network, "model_weights.json");

    // Final Memory Cleanup
    printf("Cleaning up memory...\n");
    for (int i = 0; i < DATASET_SIZE; i++) {
        free(inputTrainData[i]);
    }
    free(inputTrainData);
    free(inputLabelData);

    trainer_free(trainer);
    network_free(&network);

    printf("Done.\n");

    return EXIT_SUCCESS;
}
