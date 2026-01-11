#include "../include/relu.h"
void initReLU(struct ReLU* layer, int n_input, int h_input, int w_input, int c_input){
    layer->n_input = n_input;
    layer->h_input = h_input;
    layer->w_input = w_input;
    layer->c_input = c_input;
    layer->input = (Tensor *) malloc(sizeof(Tensor));
    initTensor(layer->input, n_input, h_input, w_input, c_input);
    layer->output = (Tensor *) malloc(sizeof(Tensor));
    initTensor(layer->output, n_input, h_input, w_input, c_input);
}
void freeReLU(struct ReLU* layer){
    freeTensor(layer->input);
    free(layer->input);
    freeTensor(layer->output);
    free(layer->output);
}
Tensor* forwardReLU(struct ReLU* layer, Tensor* input){
    assert(input->N == layer->n_input && input->H == layer->h_input &&
           input->W == layer->w_input && input->C == layer->c_input);
    for(int n = 0; n < layer->n_input; n++){
        for(int h = 0; h < layer->h_input; h++){
            for(int w = 0; w < layer->w_input; w++){
                for(int c = 0; c < layer->c_input; c++){
                    double val = input->data[n][h][w][c];
                    layer->output->data[n][h][w][c] = val > 0 ? val : 0.0;
                }
            }
        }
    }
    return layer->output;
}