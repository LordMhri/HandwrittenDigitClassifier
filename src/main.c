#include "../include/network.h"
#include <stdio.h>

int main(int argc, char const *argv[])
{
    Network network = {0};
    network_init(&network,10,10,1);
    return 0;
}
