#include "../include/network.h"
#include "../include/data_loader.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{   
    const char *inputTrainDataPath = "../dataset/train-images.idx3-ubyte";
    const char *inputLabelDataPath = "../dataset/train-labels.idx1-ubyte";
    uint8_t **inputTrainData = load_data_file(inputTrainDataPath);
    uint8_t *inputLabelData = load_text_file(inputLabelDataPath);

    shuffle(inputTrainData,inputLabelData,300);

    int num_to_load = 2;

    for (int img_idx = 0; img_idx < num_to_load; img_idx++) {
        printf("Number is %d\n",  + inputLabelData[img_idx]);

    
        // Loop through the rows and columns of each image
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                //this is flattening a 2d matrix into a 1d array
                uint8_t pixel = inputTrainData[img_idx][row * 28 + col];
                
                if (pixel > 0)
                {
                printf("\e[1;31m%3d\e[0m ", pixel); 
                } else {
                    printf("%3d ", pixel );  
                }
                
                
        }
        printf("\n"); 
    }
    printf("\n"); 
}




    // Network network = {0}; 
    // network_init(&network,10,10,1);
    // return 0;
}
