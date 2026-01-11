#ifndef POOLING_H
#define POOLING_H
#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
typedef struct PoolingLayer{
    Tensor *input;
    Tensor *output;
    int pool_size;
    int stride;
    int n_input, h_input, w_input, c_input;
    int n_output, h_output, w_output, c_output;
} PoolingLayer;
void initPoolingLayer(PoolingLayer* layer, int n_input, int h_input, int w_input, int c_input,
                      int pool_size, int stride);
void freePoolingLayer(PoolingLayer* layer);
Tensor* forwardPoolingLayer(PoolingLayer* layer, Tensor* input);
#endif