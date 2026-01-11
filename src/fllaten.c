#include "fllaten.h"
void initFlattenLayer(FlattenLayer* layer, int n_input, int h_input, int w_input, int c_input){
    layer->n_input = n_input;
    layer->h_input = h_input;
    layer->w_input = w_input;
    layer->c_input = c_input;
    layer->n_output = n_input;
    layer->h_output = 1;
    layer->w_output = 1;
    layer->c_output = h_input * w_input * c_input;
    layer->input = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->input, n_input, h_input, w_input, c_input);
    layer->output = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->output, layer->n_output, layer->h_output, layer->w_output, layer->c_output);
}
void freeFlattenLayer(FlattenLayer* layer){
    freeTensor(layer->input);
    free(layer->input);
    freeTensor(layer->output);
    free(layer->output);
}
Tensor* forwardFlattenLayer(FlattenLayer* layer, Tensor* input){
    assert(input->N == layer->n_input && input->H == layer->h_input &&
           input->W == layer->w_input && input->C == layer->c_input);
    for(int n = 0; n < layer->n_input; n++){
        int index = 0;
        for(int h = 0; h < layer->h_input; h++){
            for(int w = 0; w < layer->w_input; w++){
                for(int c = 0; c < layer->c_input; c++){
                    layer->output->data[n][0][0][index++] = input->data[n][h][w][c];
                }
            }
        }
    }
    return layer->output;
}