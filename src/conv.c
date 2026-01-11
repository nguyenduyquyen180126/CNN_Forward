#include "../include/conv.h"
void initConvLayer(ConvLayer* layer, int n_input, int h_input, int w_input, int c_input,
                   int filter_size, int out_channels, int stride, int padding){
    layer->n_input = n_input;
    layer->h_input = h_input;
    layer->w_input = w_input;
    layer->c_input = c_input;
    layer->filter_size = filter_size;
    layer->in_channels = c_input;
    layer->out_channels = out_channels;
    layer->stride = stride;
    layer->padding = padding;
    // Tính toán dim output
    layer->n_output = n_input;
    layer->h_output = (h_input - filter_size + 2 * padding) / stride + 1;
    layer->w_output = (w_input - filter_size + 2 * padding) / stride + 1;
    layer->c_output = out_channels;
    // Initialize tensors
    layer->input = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->input, n_input, h_input, w_input, c_input);
    layer->output = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->output, layer->n_output, layer->h_output, layer->w_output, layer->c_output);
    layer->filters = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->filters, out_channels, filter_size, filter_size, c_input);
    layer->biases = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->biases, 1, 1, 1, out_channels);
}
void freeConvLayer(ConvLayer* layer){
    freeTensor(layer->input);
    free(layer->input);
    freeTensor(layer->output);
    free(layer->output);
    freeTensor(layer->filters);
    free(layer->filters);
    freeTensor(layer->biases);
    free(layer->biases);
}
Tensor* forwardConvLayer(ConvLayer* layer, Tensor* input){
    for(int n = 0; n < layer->n_output; n++){
        for(int out_c = 0; out_c < layer->out_channels; out_c++){
            for(int out_h = 0; out_h < layer->h_output; out_h++){
                for(int out_w = 0; out_w < layer->w_output; out_w++){
                    double sum = 0.0;
                    for(int in_c = 0; in_c < layer->in_channels; in_c++){
                        for(int f_h = 0; f_h < layer->filter_size; f_h++){
                            for(int f_w = 0; f_w < layer->filter_size; f_w++){
                                int in_h = out_h * layer->stride + f_h - layer->padding;
                                int in_w = out_w * layer->stride + f_w - layer->padding;
                                if(in_h >= 0 && in_h < layer->h_input && in_w >= 0 && in_w < layer->w_input){
                                    sum += input->data[n][in_h][in_w][in_c] *
                                           layer->filters->data[out_c][f_h][f_w][in_c];
                                }
                            }
                        }
                    }
                    sum += layer->biases->data[0][0][0][out_c];
                    layer->output->data[n][out_h][out_w][out_c] = sum;
                }
            }
        }
    }
    return layer->output;
}
void printParam(ConvLayer* layer){
    printf("ConvLayer Parameters:\n");
    printf("Kernel:\n");
    printTensor(layer->filters);
    printf("Biases:\n");
    printTensor(layer->biases);
}
void readConvLayerParams(ConvLayer* layer, const char* filter_file, const char* bias_file){
    // Read filters
    FILE* f_file = fopen(filter_file, "r");
    if(f_file == NULL){
        printf("Error opening filter file: %s\n", filter_file);
        return;
    }
    int n, h, w, c;
    fscanf(f_file, "%d,%d,%d,%d\n", &h, &w, &c, &n);
    assert(n == layer->out_channels && h == layer->filter_size && w == layer->filter_size && c == layer->in_channels);
    for(int num_kernel = 0; num_kernel < n; num_kernel++){
        for(int channel = 0; channel < c; channel++){
            for(int height = 0; height < h; height++){
                for(int width = 0; width < w; width++){
                   if(fscanf(f_file, "%lf", &layer->filters->data[num_kernel][height][width][channel]) != 1){
                        printf("Error reading kernel value at kernel %d, channel %d, position (%d, %d)\n", num_kernel, channel, height, width);
                        fclose(f_file);
                        return;
                    }
                    // Skip comma if not the last element
                    fgetc(f_file); // lay , va xuong dong
                }
            }
        }
    }
    fclose(f_file);
    // Read biases
    FILE* b_file = fopen(bias_file, "r");
    if(b_file == NULL){
        printf("Error opening bias file: %s\n", bias_file);
        return;
    }
    int n_bias;
    fscanf(b_file, "%d\n", &n_bias);
    assert(n_bias == layer->out_channels);
    for(int i = 0; i < n_bias; i++){
        if(fscanf(b_file, "%lf", &layer->biases->data[0][0][0][i]) != 1){
            printf("Error reading bias value at index %d\n", i);
            fclose(b_file);
            return;
        }
        // Skip comma if not the last element
        fgetc(b_file); // lay , va xuong dong
    }
}