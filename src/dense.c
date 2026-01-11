#include "../include/dense.h"
// Impliment for b = 1
void initDenseLayer(DenseLayer* layer, int n_input, int n_output){
    layer->n_input = n_input;
    layer->n_output = n_output;

    layer->input = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->input, 1, 1, 1, n_input);
    layer->output = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->output, 1, 1, 1, n_output);
    layer->weights = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->weights, 1, n_input, n_output, 1);
    layer->biases = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->biases, 1, 1, 1, n_output);
}
void freeDenseLayer(DenseLayer* layer){
    freeTensor(layer->input);
    free(layer->input);
    freeTensor(layer->output);
    free(layer->output);
    freeTensor(layer->weights);
    free(layer->weights);
    freeTensor(layer->biases);
    free(layer->biases);
}
void readDenseLayerParams(DenseLayer* layer, const char* weight_file, const char* bias_file){
    FILE* file = fopen(weight_file, "r");
    if(file == NULL){
        printf("Error opening weight file: %s\n", weight_file);
        return;
    }
    int row, col;
    fscanf(file, "%d,%d\n", &row, &col);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            fscanf(file, "%lf", &layer->weights->data[0][i][j][0]);
            fgetc(file); 
        }
    }
    fclose(file);
    file = fopen(bias_file, "r");
    if(file == NULL){
        printf("Error opening bias file: %s\n", bias_file);
        return;
    }
    int n_bias;
    fscanf(file, "%d\n", &n_bias);
    for(int i = 0; i < n_bias; i++){
        fscanf(file, "%lf", &layer->biases->data[0][0][0][i]);
        fgetc(file); 
    }
}
Tensor* forwardDenseLayer(DenseLayer* layer, Tensor* input){
    assert(input->N == 1 && input->H == 1 && input->W == 1 && input->C == layer->n_input);
    // Copy input data
    for(int i = 0; i < layer->n_input; i++){
        layer->input->data[0][0][0][i] = input->data[0][0][0][i];
    }
    // Matrix multiplication and add bias
    for(int j = 0; j < layer->n_output; j++){
        double sum = 0.0;
        for(int i = 0; i < layer->n_input; i++){
            sum += layer->input->data[0][0][0][i] * layer->weights->data[0][i][j][0];
        }
        sum += layer->biases->data[0][0][0][j];
        layer->output->data[0][0][0][j] = sum;
    }
    return layer->output;
}
void softmax(DenseLayer* layer){
    for(int i = 0; i < layer->n_output; i++){
        layer->output->data[0][0][0][i] = exp(layer->output->data[0][0][0][i]);
    }
    double sum = 0.0;
    for(int i = 0; i < layer->n_output; i++){
        sum += layer->output->data[0][0][0][i];
    }
    for(int i = 0; i < layer->n_output; i++){
        layer->output->data[0][0][0][i] /= sum;
    }
}