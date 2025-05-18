#include "../include/network.h"
#include "../include/trainer.h"
#include "../include/data_loader.h"
#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>

#define LEARNING_RATE 0.1
#define BATCH_SIZE 128
#define EPOCHS 20
#define TRAIN_SAMPLES 60000
#define TARGET_ACCURACY 96.0

void mini_batch_sanity_check(Network *net, Trainer *trainer,
                             double **data, uint8_t *labels) {
    printf("\n=== Running Sanity Check (5 samples) ===\n");

    // Test mini-batch with known samples
    int test_samples = 5;
    int test_labels[5];
    double test_outputs[5][10];

    // Run forward pass before training
    printf("Pre-training predictions:\n");
    for (int i = 0; i < test_samples; i++) {
        forward_propagation(net, data[i]);
        test_labels[i] = network_predict(net);
        for (int j = 0; j < 10; j++) {
            test_outputs[i][j] = net->output_neurons[j];
        }
        printf("Sample %d: Pred=%d (True=%d)\n",
               i, test_labels[i], labels[i]);
    }

    // Train on these 5 samples
    trainer_reset_gradients(trainer);
    for (int i = 0; i < test_samples; i++) {
        forward_propagation(net, data[i]);
        backpropagate(trainer, net, data[i], labels[i]);
    }
    apply_gradients(trainer, net, test_samples,BATCH_SIZE);

    // Verify predictions changed
    printf("\nPost-training predictions:\n");
    for (int i = 0; i < test_samples; i++) {
        forward_propagation(net, data[i]);
        int new_pred = network_predict(net);
        printf("Sample %d: Pred=%d (True=%d) | Changed: %s\n",
               i, new_pred, labels[i],
               (new_pred != test_labels[i]) ? "YES" : "NO");

        // Print confidence changes
        printf("Confidence changes:\n");
        for (int j = 0; j < 10; j++) {
            printf("%d: %.2f -> %.2f (%+.2f)\n", j,
                   test_outputs[i][j],
                   net->output_neurons[j],
                   net->output_neurons[j] - test_outputs[i][j]);
        }
    }
}

int main() {
    const char *inputTrainDataPath = "../dataset/train-images.idx3-ubyte";
    const char *inputLabelDataPath = "../dataset/train-labels.idx1-ubyte";

    double **inputTrainData = load_data_file(inputTrainDataPath);
    uint8_t *inputLabelData = load_text_file(inputLabelDataPath);

    Network network = {0};
    network_init(&network, 28 * 28, 512, 128, 10);

    Trainer *trainer = trainer_init(&network, LEARNING_RATE, EPOCHS, BATCH_SIZE, TRAIN_SAMPLES, inputTrainData, inputLabelData);

    mini_batch_sanity_check(&network,trainer,inputTrainData,inputLabelData);

//    printf("--- Training with Mini-Batches (Batch Size: %d, Epochs: %d) ---\n", BATCH_SIZE, EPOCHS);
//
//    for (int epoch = 0; epoch < EPOCHS; epoch++) {
//        double total_loss = 0.0;
//        int correct_predictions = 0;
//
//        // Shuffle data at start of each epoch (optional but recommended)
//        shuffle_data(inputTrainData, inputLabelData, TRAIN_SAMPLES, 28*28);
//
//        for (int batch_start = 0; batch_start < TRAIN_SAMPLES; batch_start += BATCH_SIZE) {
//            int current_batch_size = (batch_start + BATCH_SIZE > TRAIN_SAMPLES) ?
//                                     (TRAIN_SAMPLES - batch_start) : BATCH_SIZE;
//
//            trainer_reset_gradients(trainer);
//
//            for (int i = batch_start; i < batch_start + current_batch_size; i++) {
//                forward_propagation(&network, inputTrainData[i]);
//                total_loss += cross_entropy_loss(network.output_neurons, inputLabelData[i]);
//
//                if (network_predict(&network) == inputLabelData[i]) {
//                    correct_predictions++;
//                }
//
//                backpropagate(trainer, &network, inputTrainData[i], inputLabelData[i]);
//            }
//
//            apply_gradients(trainer, &network, LEARNING_RATE, current_batch_size);
//        }
//
//        double avg_loss = total_loss / TRAIN_SAMPLES;
//        double accuracy = (double)correct_predictions / TRAIN_SAMPLES * 100.0;
//
//        printf("Epoch %2d | Loss: %.4f | Accuracy: %.2f%%\n",
//               epoch + 1, avg_loss, accuracy);
//
//        if (epoch == 0 || epoch == EPOCHS - 1) {
//            forward_propagation(&network, inputTrainData[0]);
//            printf("Sample 0 Predictions: ");
//            for (int i = 0; i < 10; i++) {
//                printf("%.4f ", network.output_neurons[i]);
//            }
//            printf("-> Pred: %d (True: %d)\n",
//                   network_predict(&network), inputLabelData[0]);
//        }
//
//        if (accuracy >= TARGET_ACCURACY) {
//            printf("Target accuracy of %.2f%% reached at epoch %d. Stopping early.\n", TARGET_ACCURACY, epoch + 1);
//            break;
//        }
//    }
//
    for (int i = 0; i < TRAIN_SAMPLES; i++) free(inputTrainData[i]);
    free(inputTrainData);
    free(inputLabelData);
    trainer_free(trainer);
    network_free(&network);

    return 0;
}



