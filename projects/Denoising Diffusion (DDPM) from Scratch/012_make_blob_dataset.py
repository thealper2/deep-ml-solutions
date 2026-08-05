import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    torch.manual_seed(seed)

    radius = size // 4
    images = torch.zeros((n, 1, size, size))

    for i in range(n):
        center_y = torch.randint(radius, size - radius, (1,)).item()
        center_x = torch.randint(radius, size - radius, (1,)).item()

        y_grid = torch.arange(size).float()
        x_grid = torch.arange(size).float()
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')

        dist = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        mask = (dist <= radius).float()
        images[i, 0] = mask

    return images
