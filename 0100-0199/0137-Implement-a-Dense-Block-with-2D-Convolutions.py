import numpy as np

def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    kh, kw = kernel_size
    N, H, W, C0 = input_data.shape
    
    pad_h = kh // 2
    pad_w = kw // 2
    
    x = input_data.copy()
    
    for l in range(num_layers):
        x_relu = np.maximum(0, x)
        
        x_padded = np.pad(x_relu, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        
        kernel = kernels[l]
        
        N, H_pad, W_pad, C_in = x_padded.shape
        H_out = H_pad - kh + 1
        W_out = W_pad - kw + 1
        
        conv_out = np.zeros((N, H_out, W_out, growth_rate))
        
        for n in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    patch = x_padded[n, h:h+kh, w:w+kw, :]
                    for c_out in range(growth_rate):
                        conv_out[n, h, w, c_out] = np.sum(patch * kernel[:, :, :, c_out])
        
        x = np.concatenate([x, conv_out], axis=-1)
    
    return x
