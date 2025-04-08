import numpy as np
import torch
from typing import List, Tuple, Union

def to_tensor(x, dtype: torch.dtype=None, device: torch.device=None) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device, dtype=dtype)
    else:
        return torch.tensor(x, dtype=dtype, device=device)

def to_batched_tensor(x: torch.tensor, batched_length: int, batch_size: int, dtype: torch.dtype=None, device: torch.device=None) -> torch.Tensor:
    x = to_tensor(x, dtype=dtype, device=device)

    if len(x.shape) != batched_length:
        # Example: A tensor with shape (1,4) and batched_length=4, batch_size=10
        #          should be expanded to (10, 1, 1, 4)
        assert batched_length > len(x.shape)
        num_missing_dims = batched_length - len(x.shape)
        x = x.reshape(*([1]*num_missing_dims), *x.shape).expand(batch_size, *([1]*(num_missing_dims-1)), *x.shape)
    
    return x

def expand_to_common_batch_size(*args, batched_lengths: Union[int, List[int], Tuple[int]], dtype: torch.dtype=torch.float32, device: torch.device=None):
    # Try to guess the device from any tensors passed as arguments
    if device is None:
        for a in args:
            if torch.is_tensor(a):
                device = device
                break

    # Convert all arguments to the tensor type
    tensors: List[torch.Tensor] = [to_tensor(a, dtype=dtype, device=device) for a in args]

    if isinstance(batched_lengths, List) or isinstance(batched_lengths, Tuple):
        assert len(tensors) == len(batched_lengths)
    else:
        batched_lengths = [batched_lengths]*len(tensors)

    # Find the first batched tensor
    idx = None
    for i, t in enumerate(tensors):
        if len(t.shape) == batched_lengths[i]:
            idx = i
            break
    
    # If there is no batched element, 
    # make the first element one with batch size 1
    is_batched = idx is not None
    if idx is None:
        idx = 0
        tensors[idx] = tensors[idx].unsqueeze(0)

    # Expand unbatched tensors to the size of the reference tensor
    tensor_ref = tensors[idx]
    for i in range(len(tensors)):
        # Skip the reference tensor
        if i == idx:
            continue
        
        # Skip batched tensors but check for inconsistencies
        if len(tensors[i].shape) == batched_lengths[i]:
            if tensors[i].shape[0] != tensor_ref.shape[0]:
                raise RuntimeError(f"Batch dimension mismatch for input: expected {tensor_ref.shape[0]} but received {tensors[i].shape[0]}.")
            continue

        tensors[i] = to_batched_tensor(tensors[i], batched_lengths[i], tensor_ref.shape[0], dtype, device)

    return *tensors, is_batched

def repeat_dim(tensor: torch.Tensor, repeats: int, dim: int):
    shape_front = tensor.shape[:dim]
    shape_back  = tensor.shape[dim+1:]
    return tensor.repeat((*[1]*len(shape_front), repeats, *[1]*len(shape_back)))

def gamma_encode(image, gamma=2.2):
    return image**(1.0/gamma)

def gamma_decode(image, gamma=2.2):
    return image**(gamma)

def vflip(image):
    """ Flip image vertically
    """
    return torch.flip(image, dims=(0,))

def crop(image, roi):
    return image[roi[0]:roi[1], roi[2]:roi[3]]

def get_roi(x, y, width, height, center_around_xy=False):
    if center_around_xy:
        return [y - height//2, y + height//2, x - width//2, x + width//2]
    else:
        return [y, y+height, x, x+height]

def to_display_image(img, flip=True, gamma=True, grayscale_to_rgb=False, to_uint8=False):
    img = img.detach().cpu()
    
    if gamma:
        img = gamma_encode(img.clamp(0, 1))

    if flip == True:
        img = vflip(img)

    if grayscale_to_rgb and img.shape[2] == 1:
        img = img.repeat((1, 1, 3))

    if to_uint8 and (img.dtype != torch.uint8):
        img = (img * 255).to(dtype=torch.uint8)

    return img

def dot(a, b, dim=-1, **kwargs):
    return torch.sum(a * b, dim=dim, **kwargs)