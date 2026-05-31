import numpy as np

def patchify_latent(latent: np.ndarray, patch_size: int, proj_weight: np.ndarray = None, proj_bias: np.ndarray = None) -> np.ndarray:
    """
    Convert a latent representation into a sequence of patch tokens
    for a Diffusion Transformer.

    Args:
        latent: Latent tensor of shape (C, H, W)
        patch_size: Size of each square patch (p)
        proj_weight: Optional projection matrix of shape (C*p*p, D)
        proj_bias: Optional projection bias of shape (D,)

    Returns:
        tokens: Array of shape (num_patches, patch_dim) or (num_patches, D)
                if projection is applied
    """
    C, H, W = latent.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_dim = C * patch_size * patch_size

    patches = np.zeros((num_patches, patch_dim))

    patch_idx = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = latent[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            flat_patch = np.array([])
            for c in range(C):
                flat_patch = np.concatenate([flat_patch, patch[c].flatten()])

            patches[patch_idx] = flat_patch
            patch_idx += 1

    if proj_weight is not None:
        patches = patches @ proj_weight
        if proj_bias is not None:
            patches += proj_bias

    return patches