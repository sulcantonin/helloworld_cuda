__global__ void norm2(float *output, float * input, int dim) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    output[id] = 0.0f;
    float val = 0;
    for(int i = 0; i<dim; i++){
        val = val + input[id * dim + i] * input[id * dim + i];
    }
    output[id] = sqrt(val);
}