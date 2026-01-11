#include "../include/tensor.h"
void initTensor(Tensor* tensor, int n, int h, int w, int c){
    tensor->N = n;
    tensor->H = h;
    tensor->W = w;
    tensor->C = c;

    tensor->data = (double ****) malloc(n * sizeof(double ***));
    for(int i=0; i<n; i++){
        tensor->data[i] = (double ***) malloc(h * sizeof(double **));
        for(int j=0; j<h; j++){
            tensor->data[i][j] = (double **) malloc(w * sizeof(double *));
            for(int k=0; k<w; k++){
                tensor->data[i][j][k] = (double *) malloc(c * sizeof(double));
            }
        }
    }
}
void freeTensor(Tensor* tensor){
    for(int i=0; i<tensor->N; i++){
        for(int j=0; j<tensor->H; j++){
            for(int k=0; k<tensor->W; k++){
                free(tensor->data[i][j][k]);
            }
            free(tensor->data[i][j]);
        }
        free(tensor->data[i]);
    }
    free(tensor->data);
}
void readTensorFromFile(Tensor* tensor, const char* filename){
    FILE* file = fopen(filename, "r");
    if(file == NULL){
        printf("Error opening file: %s\n", filename);
        return;
    }
    int file_n, file_h, file_w, file_c;
    fscanf(file, "%d,%d,%d,%d\n", &file_n, &file_h, &file_w, &file_c);
    assert(file_n == tensor->N && file_h == tensor->H && file_w == tensor->W && file_c == tensor->C);
    for(int n=0; n<tensor->N; n++){
        for(int c=0; c<tensor->C; c++){
            for(int h=0; h<tensor->H; h++){
                for(int w=0; w<tensor->W; w++){
                    fscanf(file, "%lf", &tensor->data[n][h][w][c]);
                    if(!(n == tensor->N - 1 && c == tensor->C - 1 && h == tensor->H - 1 && w == tensor->W - 1)){
                        fgetc(file); // Đọc ký tự xuống dòng hoặc dấu cách giữa các phần tử
                    }
                }
            }
        }
    }
    fclose(file);
}
void printTensor(Tensor* tensor){
    for(int n=0; n<tensor->N; n++){
        for(int c=0; c<tensor->C; c++){
            for(int h=0; h<tensor->H; h++){
                for(int w=0; w<tensor->W; w++){
                    printf("%8.4lf", tensor->data[n][h][w][c]);
                }
                printf("\n");
            }
            // printf("\n");
        }
        printf("\n");
    }
}