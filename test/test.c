#include "../include/conv.h"
#include "../include/batch_norm.h"
#include "../include/tensor.h"
#include "../include/relu.h"
#include "../include/pooling.h"
#include "../include/fllaten.h"
#include "../include/dense.h"
int main(){
    Tensor *currenrt;
    Tensor input;
    initTensor(&input, 1, 32, 32, 3);
    readTensorFromFile(&input, "input.txt");
    printTensor(&input);

    ConvLayer layer;
    initConvLayer(&layer, 1, 32, 32, 3, 3, 32, 1, 1);
    readConvLayerParams(&layer, "data/weight/conv2d_1.txt", "data/weight/conv2d_2.txt");
    // printParam(&layer);
    currenrt = forwardConvLayer(&layer, &input);
    printf("\n\n\nOutput Tensor:\n");
    printTensor(currenrt);

    BatchNormLayer bn_layer;
    initBatchNormLayer(&bn_layer, 1, 32, 32, 32, 0.9);
    readBatchNormLayerParams(&bn_layer, "data/weight/batch_normalization_1.txt", "data/weight/batch_normalization_2.txt",
                            "data/weight/batch_normalization_3.txt", "data/weight/batch_normalization_4.txt");
    // printBatchNormParam(&bn_layer);
    currenrt = forwardBatchNormLayer(&bn_layer, currenrt);
    printf("\n\n\nBatchNorm Output Tensor:\n");
    printTensor(currenrt);

    ReLU relu;
    initReLU(&relu, 1, 32, 32, 32);
    currenrt = forwardReLU(&relu, currenrt);
    printf("\n\n\nReLU Output Tensor:\n");
    printTensor(currenrt);
    
    PoolingLayer pool_layer;
    initPoolingLayer(&pool_layer, 1, 32, 32, 32, 2, 2);
    currenrt = forwardPoolingLayer(&pool_layer, currenrt);
    printf("\n\n\nPooling Output Tensor:\n");
    printTensor(currenrt);
    

    ConvLayer layer2;
    initConvLayer(&layer2, 1, 16, 16, 32, 3, 64, 1, 1);
    readConvLayerParams(&layer2, "data/weight/conv2d_1_1.txt", "data/weight/conv2d_1_2.txt");
    currenrt = forwardConvLayer(&layer2, currenrt);
    printf("\n\n\nSecond Conv Output Tensor:\n");
    printTensor(currenrt);

    BatchNormLayer bn_layer2;
    initBatchNormLayer(&bn_layer2, 1, 16, 16, 64, 0.9);
    readBatchNormLayerParams(&bn_layer2, "data/weight/batch_normalization_1_1.txt", "data/weight/batch_normalization_1_2.txt",
                            "data/weight/batch_normalization_1_3.txt", "data/weight/batch_normalization_1_4.txt");
    currenrt = forwardBatchNormLayer(&bn_layer2, currenrt);
    printf("\n\n\nSecond BatchNorm Output Tensor:\n");
    printTensor(currenrt);

    ReLU relu2;
    initReLU(&relu2, 1, 16, 16, 64);
    currenrt = forwardReLU(&relu2, currenrt);
    printf("\n\n\nSecond ReLU Output Tensor:\n");
    printTensor(currenrt);

    PoolingLayer pool_layer2;
    initPoolingLayer(&pool_layer2, 1, 16, 16, 64, 2, 2);
    currenrt = forwardPoolingLayer(&pool_layer2, currenrt);
    printf("\n\n\nSecond Pooling Output Tensor:\n");
    printTensor(currenrt);

    FlattenLayer flatten_layer;
    initFlattenLayer(&flatten_layer, 1, 8, 8, 64);
    currenrt = forwardFlattenLayer(&flatten_layer, currenrt);
    printf("\n\n\nFlatten Output Tensor:\n");
    printTensor(currenrt);

    DenseLayer dense_layer;
    initDenseLayer(&dense_layer, 8*8*64, 128);
    readDenseLayerParams(&dense_layer, "data/weight/dense_1.txt", "data/weight/dense_2.txt");
    currenrt = forwardDenseLayer(&dense_layer, currenrt);
    printf("dense weights:\n");
    printTensor(dense_layer.weights);
    printf("dense biases:\n");
    printTensor(dense_layer.biases);
    printf("\n\n\nDense Output Tensor:\n");
    printTensor(currenrt);

    BatchNormLayer bn_layer3;
    initBatchNormLayer(&bn_layer3, 1, 1, 1, 128, 0.9);
    readBatchNormLayerParams(&bn_layer3, "data/weight/batch_normalization_2_1.txt", "data/weight/batch_normalization_2_2.txt",
                            "data/weight/batch_normalization_2_3.txt", "data/weight/batch_normalization_2_4.txt");
    currenrt = forwardBatchNormLayer(&bn_layer3, currenrt);
    printf("\n\n\nThird BatchNorm Output Tensor:\n");
    printTensor(currenrt);

    ReLU relu3;
    initReLU(&relu3, 1, 1, 1, 128);
    currenrt = forwardReLU(&relu3, currenrt);
    printf("\n\n\nThird ReLU Output Tensor:\n");
    printTensor(currenrt);

    DenseLayer dense_layer2;
    initDenseLayer(&dense_layer2, 128, 64);
    readDenseLayerParams(&dense_layer2, "data/weight/dense_1_1.txt", "data/weight/dense_1_2.txt");
    currenrt = forwardDenseLayer(&dense_layer2, currenrt);
    printf("\n\n\nFinal Dense Output Tensor:\n");
    printTensor(currenrt);

    BatchNormLayer bn_layer4;
    initBatchNormLayer(&bn_layer4, 1, 1, 1, 64, 0.9);
    readBatchNormLayerParams(&bn_layer4, "data/weight/batch_normalization_3_1.txt", "data/weight/batch_normalization_3_2.txt",
                            "data/weight/batch_normalization_3_3.txt", "data/weight/batch_normalization_3_4.txt");
    currenrt = forwardBatchNormLayer(&bn_layer4, currenrt);
    printf("\n\n\nFourth BatchNorm Output Tensor:\n");
    printTensor(currenrt);

    ReLU relu4;
    initReLU(&relu4, 1, 1, 1, 64);
    currenrt = forwardReLU(&relu4, currenrt);
    printf("\n\n\nFinal ReLU Output Tensor:\n");
    printTensor(currenrt);

    DenseLayer dense_layer3;
    initDenseLayer(&dense_layer3, 64, 10);
    readDenseLayerParams(&dense_layer3, "data/weight/dense_2_1.txt", "data/weight/dense_2_2.txt");
    currenrt = forwardDenseLayer(&dense_layer3, currenrt);
    printf("\n\n\nFinal Output Tensor:\n");
    printTensor(currenrt);
    printf("\n\n\nSoftmax Output Tensor:\n");
    softmax(&dense_layer3);
    printTensor(currenrt);
    

    freeFlattenLayer(&flatten_layer);
    freeConvLayer(&layer);
    freeBatchNormLayer(&bn_layer);
    freeReLU(&relu);
    freeTensor(&input);
    freePoolingLayer(&pool_layer);
    freeConvLayer(&layer2);
    freeBatchNormLayer(&bn_layer2);
    freeReLU(&relu2);
    freePoolingLayer(&pool_layer2);
    freeDenseLayer(&dense_layer);
    freeBatchNormLayer(&bn_layer3);
    freeReLU(&relu3);
    freeDenseLayer(&dense_layer2);
    freeBatchNormLayer(&bn_layer4);
    freeReLU(&relu4);
    freeDenseLayer(&dense_layer3);
    return 0;
}