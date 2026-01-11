#include "pooling.h"
void initPoolingLayer(PoolingLayer* layer, int n_input, int h_input, int w_input, int c_input,
                      int pool_size, int stride){
    layer->n_input = n_input;
    layer->h_input = h_input;
    layer->w_input = w_input;
    layer->c_input = c_input;
    layer->pool_size = pool_size;
    layer->stride = stride;
    layer->n_output = n_input;
    layer->h_output = (h_input - pool_size) / stride + 1;
    layer->w_output = (w_input - pool_size) / stride + 1;
    layer->c_output = c_input;
    layer->input = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->input, n_input, h_input, w_input, c_input);
    layer->output = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->output, layer->n_output, layer->h_output, layer->w_output, c_input);
}
void freePoolingLayer(PoolingLayer* layer){
    freeTensor(layer->input);
    free(layer->input);
    freeTensor(layer->output);
    free(layer->output);
}
Tensor* forwardPoolingLayer(PoolingLayer* layer, Tensor* input){
    assert(input->N == layer->n_input && input->H == layer->h_input &&
           input->W == layer->w_input && input->C == layer->c_input);
    for(int n = 0; n < layer->n_input; n++){
        for(int c = 0; c < layer->c_input; c++){
            for(int h_out = 0; h_out < layer->h_output; h_out++){
                for(int w_out = 0; w_out < layer->w_output; w_out++){
                    double max_val = -1e10;
                    for(int ph = 0; ph < layer->pool_size; ph++){
                        for(int pw = 0; pw < layer->pool_size; pw++){
                            int h_in = h_out * layer->stride + ph;
                            int w_in = w_out * layer->stride + pw;
                            if(h_in < layer->h_input && w_in < layer->w_input){
                                double val = input->data[n][h_in][w_in][c];
                                if(val > max_val){
                                    max_val = val;
                                }
                            }
                        }
                    }
                    layer->output->data[n][h_out][w_out][c] = max_val;
                }
            }
        }
    }
    return layer->output;
}