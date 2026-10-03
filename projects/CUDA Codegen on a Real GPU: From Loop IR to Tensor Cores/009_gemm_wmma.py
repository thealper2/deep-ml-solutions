def gemm_wmma(M, N, K):
    assert M % 16 == 0 and N % 16 == 0 and K % 16 == 0
    warps = (M // 16) * (N // 16)
    return f"""__global__ void gemm_wmma(float* data0, const __half* data1, const __half* data2) {{
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int ti = warp / {N // 16};
  int tj = warp % {N // 16};
  if (ti >= {M // 16}) return;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  for (int k = 0; k < {K}; k += 16) {{
    wmma::load_matrix_sync(a_frag, data1 + ti * 16 * {K} + k, {K});
    wmma::load_matrix_sync(b_frag, data2 + k * {N} + tj * 16, {N});
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }}
  wmma::store_matrix_sync(data0 + ti * 16 * {N} + tj * 16, c_frag, {N}, wmma::mem_row_major);
}}
extern "C" void launch_gemm_wmma(float* data0, const __half* data1, const __half* data2) {{
  int warps = {warps};
  gemm_wmma<<<(warps * 32 + 127) / 128, 128>>>(data0, data1, data2);
  cudaDeviceSynchronize();
}}
"""