#ifndef RELU_H
#define RELU_H
#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
struct ReLU{
    Tensor *input;
    Tensor *output;
    int n_input, h_input, w_input, c_input;
};
void initReLU(struct ReLU* layer, int n_input, int h_input, int w_input, int c_input);
void freeReLU(struct ReLU* layer);
Tensor* forwardReLU(struct ReLU* layer, Tensor* input);
#endif