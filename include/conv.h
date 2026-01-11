#ifndef CONV_H
#define CONV_H
#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
typedef struct ConvLayer{
    Tensor *input;
    Tensor *output;
    Tensor *filters;
    Tensor *biases;
    int filter_size;
    int in_channels;
    int out_channels;
    int stride;
    int padding;
    int n_input, h_input, w_input, c_input;
    int n_output, h_output, w_output, c_output;
} ConvLayer;
void initConvLayer(ConvLayer* layer, int n_input, int h_input, int w_input, int c_input,
                   int filter_size, int out_channels, int stride, int padding);
void freeConvLayer(ConvLayer* layer);
void readConvLayerParams(ConvLayer* layer, const char* filter_file, const char* bias_file);
Tensor* forwardConvLayer(ConvLayer* layer, Tensor* input);
void printParam(ConvLayer* layer);
#endif