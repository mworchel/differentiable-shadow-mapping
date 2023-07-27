import math
import torch
from typing import List, Union

from .common import expand_to_common_batch_size, to_batched_tensor

def homogeneous(tensor):
    return torch.nn.functional.pad(tensor, pad=(0, 1), mode='constant', value=1)

def hnormalized(tensor):
    return tensor[..., :-1] / tensor[..., -1:]

def create_perspective_projection_matrix(fovy: Union[float, torch.Tensor] = 90, aspect: Union[float, torch.Tensor] = 1, near: Union[float, torch.Tensor] = 0.1, far: Union[float, torch.Tensor] = 10.0, device: torch.device = None):
    """ Create a perspective projection matrix. 
    
    Returns an array of matrices if arguments are arrays of values.
    
    Args:
        fovy: Horizontal field of view in degrees
        aspect: Aspect ratio
        near: Near plane distance
        far: Far plane distance
    """

    fovy, aspect, near, far, is_batched = expand_to_common_batch_size(fovy, aspect, near, far, batched_lengths=1, dtype=torch.float32, device=device)
    
    tan_half_fovy = torch.tan(0.5*torch.deg2rad(fovy))

    P = torch.zeros((4, 4), dtype=torch.float32, device=device)[None].repeat(fovy.shape[0], 1, 1)
    P[:, 0, 0] = 1/(aspect*tan_half_fovy)
    P[:, 1, 1] = 1/tan_half_fovy
    P[:, 2, 2] = -(far+near)/(far-near)
    P[:, 2, 3] = -(2*far*near)/(far-near)
    P[:, 3, 2] = -1

    return P if is_batched else P[0]

def create_orthographic_projection_matrix(size: Union[torch.Tensor, float] = 1, near: Union[torch.Tensor, float] = 0.1, far: Union[torch.Tensor, float] = 10.0, device: torch.device = None):
    """ Create an orthographic projection matrix. 
    
    Returns an array of matrices if arguments are arrays of values.
    
    Args:
        size: Size of the sensor in world coordinates
        near: Near plane distance
        far:  Far plane distance
    """
    size, near, far, is_batched = expand_to_common_batch_size(size, near, far, batched_lengths=1, dtype=torch.float32, device=device)

    P = torch.eye(4, dtype=torch.float32, device=device)[None].repeat(size.shape[0], 1, 1)
    P[:, 0, 0] = 1/size
    P[:, 1, 1] = 1/size
    P[:, 2, 2] = -2/(far-near)
    P[:, 2, 3] = -(far + near)/(far - near)

    return P if is_batched else P[0]

def create_translation_matrix(x: Union[torch.Tensor, float], y: Union[torch.Tensor, float], z: Union[torch.Tensor, float], device: torch.device=None):
    """ Create a translation matrix. 
    
    Returns an array of matrices if arguments are arrays of values.
    
    Args:
        x: Translation in x direction
        y: Translation in y direction
        z: Translation in z direction
    """
    x, y, z, is_batched = expand_to_common_batch_size(x, y, z, batched_lengths=1, dtype=torch.float32, device=device)

    T = torch.eye(4, dtype=torch.float32, device=device)[None].repeat(x.shape[0], 1, 1)
    T[:, 0, 3] = x
    T[:, 1, 3] = y
    T[:, 2, 3] = z

    return T if is_batched else T[0]

def create_scale_matrix(sx: Union[torch.Tensor, float], sy: Union[torch.Tensor, float], sz: Union[torch.Tensor, float], device: torch.device=None):
    """ Create a scaling matrix. 
    
    Returns an array of matrices if arguments are arrays of values.
    
    Args:
        sx: Scaling in x direction
        sy: Scaling in y direction
        sz: Scaling in z direction
    """

    sx, sy, sz, is_batched = expand_to_common_batch_size(sx, sy, sz, batched_lengths=1, dtype=torch.float32, device=device)

    S = torch.eye(4, dtype=torch.float32, device=device)[None].repeat(sx.shape[0], 1, 1)
    S[:, 0, 0] = sx
    S[:, 1, 1] = sy
    S[:, 2, 2] = sz

    return S if is_batched else S[0]
    
def create_rotation_matrix_y(angle: float, device: torch.device = None):
    """ Create a 4x4 matrix for a rotation in the ij-plane 

    Note: Deprecated in favor of `create_rotation_matrix`

    Args:
        angle: The rotation angle in radians
        device: The device to create the rotation matrix on
    """

    if torch.is_tensor(angle):
        s, c = torch.sin(angle), torch.cos(angle)
    else:
        s, c = math.sin(angle), math.cos(angle)

    R = torch.eye(4, dtype=torch.float32, device=device)
    R[0, 0] = c
    R[2, 2] = c
    R[0, 2] = s
    R[2, 0] = -s
    
    return R

def create_rotation_matrix(angle: float, i: int, j: int, device: torch.device = None):
    """ Create a 4x4 matrix for a rotation in the ij-plane 

    Args:
        angle: The rotation angle in radians
        i: Index of the 1st axis defining the rotation plane (x=0, y=1, z=2)
        j: Index of the 2nd axis defining the rotation plane (x=0, y=1, z=2)
        device: The device to create the rotation matrix on
    """

    i, j = min(i, j), max(i, j)

    if torch.is_tensor(angle):
        s, c = torch.sin(angle), torch.cos(angle)
    else:
        s, c = math.sin(angle), math.cos(angle)

    R = torch.eye(4, dtype=torch.float32, device=device)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = s
    R[j, i] = -s
    
    return R

def create_lookat_matrix(eye: Union[List[float], torch.Tensor], focus: Union[List[float], torch.Tensor], up: Union[List[float], torch.Tensor], device: torch.device = None):
    """ Create a look-at view matrix (OpenGL conventions)
    
    Returns an array of matrices if arguments are arrays of values.

    Args:
        eye: Position of the camera in world coordinates
        focus: Focus point in world coordinates
        up: Camera up-direction in world coordinates
    """

    eye, focus, up, is_batched = expand_to_common_batch_size(eye, focus, up, batched_lengths=[2, 2, 2], dtype=torch.float32, device=device)

    z = eye - focus
    x = torch.linalg.cross(up, z, dim=-1)
    x_backup = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=eye.device)[None].expand_as(x)
    x_length = torch.linalg.norm(x, dim=-1, keepdim=True)
    x = torch.where(torch.isclose(x_length, torch.zeros_like(x_length)), x_backup, x)
    y = torch.linalg.cross(z, x, dim=-1)

    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    y = torch.nn.functional.normalize(y, p=2, dim=-1)
    z = torch.nn.functional.normalize(z, p=2, dim=-1)

    R = torch.eye(4, dtype=torch.float32, device=eye.device)[None].repeat(x.shape[0], 1, 1)
    R[:, :3, :3] = torch.stack([x, y, z], dim=1)

    T = create_translation_matrix(-eye[:, 0], -eye[:, 1], -eye[:, 2], device=eye.device)

    RT = torch.bmm(R, T)

    return RT if is_batched else RT[0]

def create_view_matrix_from_direction(direction: Union[List[float], torch.Tensor], distance: Union[float, torch.Tensor], device: torch.device = None):
    """ Create a view matrix for a camera observing along a given direction (OpenGL conventions)
    
    Returns an array of matrices if arguments are arrays of values.

    Args:
        direction: Direction indicating the (inverse) optical axis (i.e., pointing towards the camera)
        distance: Distance of the camera center from the origin
    """

    direction, distance, is_batched = expand_to_common_batch_size(direction, distance, batched_lengths=[2, 1], dtype=torch.float32, device=device)
    
    device = direction.device

    cz = torch.nn.functional.normalize(-direction, p=2, dim=-1)

    up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)[None].expand_as(cz)
    cx = torch.linalg.cross(up, cz, dim=-1)

    if torch.isclose(torch.linalg.norm(cx), torch.tensor(0.0, device=device)):
        # TODO: Choose more reasonable fallback?
        cx = torch.tensor([[1, 0, 0]], dtype=torch.float32, device=device).expand_as(cx)
    else:
        cx = torch.nn.functional.normalize(cx, p=2, dim=-1)

    cy = torch.linalg.cross(cz, cx, dim=-1)

    R = torch.eye(4, dtype=torch.float32, device=device)[None].repeat(cx.shape[0], 1, 1)
    R[:, :3, :3] = torch.stack([cx, cy, cz], dim=1)

    T = create_translation_matrix(0, 0, -distance, device=device)

    # TODO: Check if this works as intended if T and R are batched
    V = T @ R    

    return V if is_batched else V[0]

def create_cubemap_view_matrix(position: torch.Tensor, id):
    position = to_batched_tensor(position, batched_length=2, batch_size=1, dtype=torch.float32)

    focus, up = {
        0: ([1, 0, 0],  [0, -1, 0]),
        1: ([-1, 0, 0], [0, -1, 0]),
        2: ([0, 1, 0],  [0, 0, 1]),  # Attention with the up direction!
        3: ([0, -1, 0], [0, 0, -1]), #
        4: ([0, 0, 1],  [0, -1, 0]),
        5: ([0, 0, -1], [0, -1, 0]),
    }[id]
    
    focus = to_batched_tensor(focus, batched_length=2, batch_size=position.shape[0], dtype=torch.float32, device=position.device)
    up    = to_batched_tensor(up,    batched_length=2, batch_size=position.shape[0], dtype=torch.float32, device=position.device)

    return create_lookat_matrix(eye=position, focus=position+focus, up=up)

def apply_transformation(positions: torch.tensor, matrix: torch.tensor, mode='projective'):
    """ Apply a linear transformation to an array of positions.

    Note: For a projective transformation, the result is returned in homogeneous coordinates

    Args:
        positions: Positions to be transformed with shape (...,3)
        matrix: Transformation matrix with shape          (4,4)
        mode: Mode for interpreting the transformation matrix (euclidean, affine, projective)
    """

    if mode == 'projective':
        positions_homogeneous = torch.cat([positions, torch.ones_like(positions[..., :1])], dim=-1)
        return positions_homogeneous @ matrix.T
    elif mode == 'affine':
        # FIXME: Correctly unsqueeze dimensions of the translation vector
        return positions @ matrix[:3, :3].T + matrix[:3, 3]
    elif mode == 'euclidean':
        return positions @ matrix[:3, :3].T
    else:
        raise RuntimeError(f"Unknown mode '{mode}'.")