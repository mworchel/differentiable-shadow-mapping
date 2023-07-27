import torch

def make_grid(sizes, limits=(-1, 1), device: torch.device = None):
    # Check if limits are intended for all dimensions
    if len(limits) == 2 and not hasattr(limits[0], '__len__'):
        limits = [limits]*len(sizes)

    # Flip the y-axis for images
    if len(sizes) == 2:
        limits[1] = (limits[1][1], limits[1][0])

    xs = []
    for size_x, limits_x in zip(sizes, limits):
        xs += [ torch.linspace(*limits_x, size_x, device=device) ]

    return torch.stack(torch.meshgrid(*xs[::-1], indexing='ij')[::-1], dim=-1)

def get_box_filter_1d(size):
    kernel = torch.ones((1, 1, size), dtype=torch.float32)
    return kernel / size

def get_gaussian_filter_1d(size):
    x = torch.linspace(-1, 1, size) 
    kernel = torch.exp(-x**2)
    kernel /= kernel.sum()
    return kernel.view(1, 1, -1)

def get_wendland_filter_1d(size):
    x = torch.linspace(0, 1, size//2 + 1)
    kernel = (1-x)**4 * (4*x + 1)
    kernel = torch.cat([kernel[1:].flip(0), kernel])
    kernel /= kernel.sum()
    return kernel.view(1, 1, -1)

def get_box_filter_2d(size: int, device: torch.device = None):
    kernel = torch.ones((1, 1, size, size), dtype=torch.float32, device=device)
    kernel /= kernel.shape[2]*kernel.shape[3]
    return kernel

def get_gaussian_filter_2d(size: int, device: torch.device = None):
    xy = make_grid((size, size), device=device)
    kernel = torch.exp(-(xy[..., 0]**2 + xy[..., 1]**2)/(2*0.1))
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, size, size)
    return kernel

def get_wendland_filter_2d(size: int, device: torch.device = None):
    xy = make_grid((size, size), device=device)
    h = 1 #math.sqrt(2)
    d = torch.linalg.norm(xy, dim=-1).clamp(max=h)
    kernel = (1-d/h)**4 * (4*d/h + 1)
    kernel /= kernel.sum()
    return kernel.view(1, 1, size, size)

def apply_filter_2d(image: torch.Tensor, kernel: torch.Tensor, padding_mode: str) -> torch.Tensor:
    """ Convolve an all channels of an image (or batch of images) with a filter kernel
    
    Args:
        image: The set of images  ((B,)H,W,C)
        kernel: The filter kernel ((1, 1,)KH,KW)
        padding_mode: Padding mode (see `torch.nn.functional.pad`)
    """

    # Convert inputs to the required shape
    is_batched = len(image.shape) == 4
    image  = image if is_batched else image[None]
    kernel = kernel if len(kernel.shape) == 4 else kernel[None, None]

    assert len(image.shape) == 4, "image must have shape [>0, >0, >0, >0]"
    assert len(kernel.shape) == 4, "kernel must have shape [>0, >0, >0, >0]"

    padding      = (kernel.shape[2]//2, kernel.shape[2]//2, kernel.shape[2]//2, kernel.shape[2]//2)
    image_padded = torch.nn.functional.pad(image.permute(0, 3, 1, 2), padding, mode=padding_mode)

    image_filtered = torch.nn.functional.conv2d(image_padded, kernel).permute(0, 2, 3, 1)

    return image_filtered if is_batched else image_filtered[0]