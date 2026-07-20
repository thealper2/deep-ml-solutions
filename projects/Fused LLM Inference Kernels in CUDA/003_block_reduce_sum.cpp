__device__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float sum = 0.0f;
        int num_warps = (blockDim.x + 31) >> 5;
        if (lane < num_warps) {
            sum = shared[lane];
        }
        sum = warp_reduce_sum(sum);
        shared[0] = sum;
    }
    __syncthreads();
    
    return (tid == 0) ? shared[0] : 0.0f;
}
