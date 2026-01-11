#include "batch_norm.h"
void initBatchNormLayer(BatchNormLayer* layer, int n_input, int h_input, int w_input, int c_input, double momentum){
    layer->n_input = n_input;
    layer->h_input = h_input;
    layer->w_input = w_input;
    layer->c_input = c_input;
    layer->momentum = momentum;

    layer->input = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->input, n_input, h_input, w_input, c_input);
    layer->output = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->output, n_input, h_input, w_input, c_input);
    layer->gamma = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->gamma, 1, 1, 1, c_input);
    layer->beta = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->beta, 1, 1, 1, c_input);
    layer->running_mean = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->running_mean, 1, 1, 1, c_input);
    layer->running_var = (Tensor*) malloc(sizeof(Tensor));
    initTensor(layer->running_var, 1, 1, 1, c_input);
}
void freeBatchNormLayer(BatchNormLayer* layer){
    freeTensor(layer->input);
    free(layer->input);
    freeTensor(layer->output);
    free(layer->output);
    freeTensor(layer->gamma);
    free(layer->gamma);
    freeTensor(layer->beta);
    free(layer->beta);
    freeTensor(layer->running_mean);
    free(layer->running_mean);
    freeTensor(layer->running_var);
    free(layer->running_var);
}
void readBatchNormLayerParams(BatchNormLayer* layer, const char* gamma_file, const char* beta_file,
                             const char* running_mean_file, const char* running_var_file){
    FILE* file = fopen(gamma_file, "r");
    if(file == NULL){
        printf("Error opening gamma file: %s\n", gamma_file);
        return;
    }
    // Read gamma values
    fscanf(file, "%d\n", &layer->c_input);
    for(int c = 0; c < layer->c_input; c++){
        fscanf(file, "%lf", &layer->gamma->data[0][0][0][c]);
        fgetc(file); // consume newline
    }
    fclose(file);

    file = fopen(beta_file, "r");
    if(file == NULL){
        printf("Error opening beta file: %s\n", beta_file);
        return;
    }
    // Read beta values
    fscanf(file, "%d\n", &layer->c_input);
    for(int c = 0; c < layer->c_input; c++){
        fscanf(file, "%lf", &layer->beta->data[0][0][0][c]);
        fgetc(file); // consume newline
    }
    fclose(file);
    file = fopen(running_mean_file, "r");
    if(file == NULL){
        printf("Error opening running mean file: %s\n", running_mean_file);
        return;
    }
    // Read running mean values
    fscanf(file, "%d\n", &layer->c_input);
    for(int c = 0; c < layer->c_input; c++){
        fscanf(file, "%lf", &layer->running_mean->data[0][0][0][c]);
        fgetc(file); // consume newline
    }
    fclose(file);
    file = fopen(running_var_file, "r");
    if(file == NULL){
        printf("Error opening running variance file: %s\n", running_var_file);
        return;
    }
    // Read running variance values
    fscanf(file, "%d\n", &layer->c_input);
    for(int c = 0; c < layer->c_input; c++){
        fscanf(file, "%lf", &layer->running_var->data[0][0][0][c]);
        fgetc(file); // consume newline
    }
    fclose(file);

}
Tensor* forwardBatchNormLayer(BatchNormLayer* layer, Tensor* input){
    if(input->H > 0 && input->W > 0){
        for(int n=0; n<layer->n_input; n++){
            for(int c=0; c<layer->c_input; c++){
                double mean = layer->running_mean->data[0][0][0][c];
                double var = layer->running_var->data[0][0][0][c];
                double gamma = layer->gamma->data[0][0][0][c];
                double beta = layer->beta->data[0][0][0][c];
                for(int h=0; h<layer->h_input; h++){
                    for(int w=0; w<layer->w_input; w++){
                        double x = input->data[n][h][w][c];
                        double x_hat = (x - mean) / sqrt(var + 0.001);
                        layer->output->data[n][h][w][c] = gamma * x_hat + beta;
                    }
                }
            }
        }
        return layer->output;
    } 
    else if(input->H == 1 && input->W == 1){
        for(int n=0; n<layer->n_input; n++){
            for(int c=0; c<layer->c_input; c++){
                double mean = layer->running_mean->data[0][0][0][c];
                double var = layer->running_var->data[0][0][0][c];
                double gamma = layer->gamma->data[0][0][0][c];
                double beta = layer->beta->data[0][0][0][c];
                double x = input->data[n][0][0][c];
                double x_hat = (x - mean) / sqrt(var + 0.001);
                layer->output->data[n][0][0][c] = gamma * x_hat + beta;
            }
        }
        return layer->output;
    }
    return NULL;
}
void printBatchNormParam(BatchNormLayer* layer){
    printf("BatchNorm Layer Parameters:\n");
    printf("Gamma:\n");
    printTensor(layer->gamma);
    printf("Beta:\n");
    printTensor(layer->beta);
    printf("Running Mean:\n");
    printTensor(layer->running_mean);
    printf("Running Variance:\n");
    printTensor(layer->running_var);
}