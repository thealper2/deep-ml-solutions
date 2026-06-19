def output_spatial_size(input_size, kernel, stride, padding):
    h_out = (input_size + 2 * padding - kernel) // stride + 1
    return h_out
