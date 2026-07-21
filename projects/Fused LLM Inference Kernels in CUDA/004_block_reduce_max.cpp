__device__ float block_reduce_max(float val, float* shared) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = -INFINITY;
        int num_warps = (blockDim.x + 31) >> 5;
        if (lane < num_warps) {
            max_val = shared[lane];
        }
        max_val = warp_reduce_max(max_val);
        shared[0] = max_val;
    }
    __syncthreads();

    return (tid == 0) ? shared[0] : 0.0f;   
}
