import numpy as np

def patch_embed_reconstruct(image: np.ndarray, patch_size: int):
    """
    Split image into patches, flatten to patch embedding matrix,
    then reconstruct the original image.

    Args:
        image: 2D numpy array of shape (H, W)
        patch_size: int, side length of each square patch

    Returns:
        Reconstructed image as a nested list of shape (H, W),
        or -1 if dimensions are invalid.
    """
    H, W = image.shape
    if H % patch_size != 0 or W % patch_size != 0:
        return -1

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    N = num_patches_h * num_patches_w

    patches = image.reshape(num_patches_h, patch_size, num_patches_w, patch_size)
    patches = patches.transpose(0, 2, 1, 3)

    embedding_matrix = patches.reshape(N, patch_size * patch_size)

    reconstructed_patches = embedding_matrix.reshape(num_patches_h, num_patches_w, patch_size, patch_size)
    reconstructed_patches = reconstructed_patches.transpose(0, 2, 1, 3)
    reconstructed_image = reconstructed_patches.reshape(H, W)
    return reconstructed_image.tolist()