#ifndef TENSOR_H
#define TENSOR_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
typedef struct Tensor{
    double ****data;
    int N, H, W, C; // Number, Height, Width, Channel
} Tensor;
void initTensor(Tensor* tensor, int n, int h, int w, int c);
void freeTensor(Tensor* tensor);
void readTensorFromFile(Tensor* tensor, const char* filename);
void printTensor(Tensor* tensor);
#endif