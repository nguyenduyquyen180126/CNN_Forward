#ifndef FLLATEN_H
#define FLLATEN_H
#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
typedef struct FlattenLayer{
    Tensor *input;
    Tensor *output;
    int n_input, h_input, w_input, c_input;
    int n_output, h_output, w_output, c_output;
} FlattenLayer;
void initFlattenLayer(FlattenLayer* layer, int n_input, int h_input, int w_input, int c_input);
void freeFlattenLayer(FlattenLayer* layer);
Tensor* forwardFlattenLayer(FlattenLayer* layer, Tensor* input);
#endif