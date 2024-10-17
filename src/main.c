#include "../include/network.h"
#include "../include/data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BATCH_SIZE 32
#define LEARNING_RATE 0.01
#define EPOCHS 10

int main(int argc, char const *argv[]) {
    const char *inputTrainDataPath = "../dataset/train-images.idx3-ubyte";
    const char *inputLabelDataPath = "../dataset/train-labels.idx1-ubyte";
    uint8_t **inputTrainData = load_data_file(inputTrainDataPath);
    uint8_t *inputLabelData = load_text_file(inputLabelDataPath);
    
    srand(time(NULL));
    int dataset_size = 60000;
    shuffle(inputTrainData, inputLabelData, dataset_size);

    // int num_to_load = 100;
    // for (int img_idx = 0; img_idx < num_to_load; img_idx++) {
    //     printf("Number is %d\n", inputLabelData[img_idx]);
    //     // Loop through the rows and columns of each image
    //     for (int row = 0; row < 28; row++) {
    //         for (int col = 0; col < 28; col++) {
    //             // Flattening a 2D matrix into a 1D array
    //             uint8_t pixel = inputTrainData[img_idx][row * 28 + col];
    //             if (pixel > 0) {
    //                 printf("\e[1;31m%3d\e[0m ", pixel);
    //             } else {
    //                 printf("%3d ", pixel);
    //             }
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // Initialize network
    Network network = {0};
    network_init(&network, 28 * 28, 64, 10);

    // Initialize trainer
    Trainer trainer = {0};
    trainer_init(&trainer, &network);

    // Train the model
    trainer_Mini_Batch_train(&trainer, &network, inputTrainData, inputLabelData, EPOCHS, BATCH_SIZE, LEARNING_RATE, dataset_size);

    // Free the network and trainer
    network_free(&network);
    trainer_free(&trainer);

    // Free input data
    for (int i = 0; i < dataset_size; i++) {
        free(inputTrainData[i]);
    }
    free(inputTrainData);
    free(inputLabelData);

    return 0;
}