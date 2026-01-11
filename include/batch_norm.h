#ifndef BATCH_NORM_H
#define BATCH_NORM_H
#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
typedef struct BatchNormLayer{
    Tensor *input;
    Tensor *output;
    Tensor *gamma;
    Tensor *beta;
    Tensor *running_mean;
    Tensor *running_var;
    double momentum;
    int n_input, h_input, w_input, c_input;
} BatchNormLayer;
void initBatchNormLayer(BatchNormLayer* layer, int n_input, int h_input, int w_input, int c_input, double momentum);
void freeBatchNormLayer(BatchNormLayer* layer);
void readBatchNormLayerParams(BatchNormLayer* layer, const char* gamma_file, const char* beta_file,
                             const char* running_mean_file, const char* running_var_file);
Tensor* forwardBatchNormLayer(BatchNormLayer* layer, Tensor* input);
void printBatchNormParam(BatchNormLayer* layer);
#endif