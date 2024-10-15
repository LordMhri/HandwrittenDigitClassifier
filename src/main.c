#include "../include/network.h"
#include "../include/data_loader.h"
#include <stdio.h>

int main(int argc, char const *argv[])
{   
    const char *inputTrainDataPath = "../dataset/train-images.idx3-ubyte";
    const char *inputLabelDataPath = "../dataset/train-labels.idx1-ubyte";
    uint8_t **inputTrainData = load_data_file(inputTrainDataPath);
    uint8_t *inputLabelData = load_text_file(inputLabelDataPath);
    Network network = {0}; 
    network_init(&network,10,10,1);
    return 0;
}
