def im2col(images, kernel_h, kernel_w, stride, padding):
    padded = pad_2d(images, padding)
    N, C, H_pad, W_pad = padded.shape

    H_out = output_spatial_size(images.shape[2], kernel_h, stride, padding)
    W_out = output_spatial_size(images.shape[3], kernel_w, stride, padding)

    patch_size = C * kernel_h * kernel_w
    num_patches = N * H_out * W_out
    patches = np.zeros((num_patches, patch_size), dtype=images.dtype)

    patch_idx = 0
    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * stride
                h_end = h_start + kernel_h
                w_start = w_out * stride
                w_end = w_start + kernel_w

                patch = padded[n, :, h_start:h_end, w_start:w_end]
                patches[patch_idx] = patch.flatten()
                patch_idx += 1

    return patches
