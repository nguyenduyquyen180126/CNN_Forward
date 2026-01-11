#ifndef DENSE_H
#define DENSE_H
#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
typedef struct DenseLayer{
    Tensor *input;
    Tensor *output;
    Tensor *weights;
    Tensor *biases;
    int n_input;
    int n_output;
} DenseLayer;
void initDenseLayer(DenseLayer* layer, int n_input, int n_output);
void freeDenseLayer(DenseLayer* layer);
void readDenseLayerParams(DenseLayer* layer, const char* weight_file, const char* bias_file);
Tensor* forwardDenseLayer(DenseLayer* layer, Tensor* input);
void softmax(DenseLayer* layer);
#endif