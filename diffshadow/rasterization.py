
from dataclasses import dataclass, field
import torch

from .transformation import homogeneous

@dataclass
class PreRasterizationData:
    is_instanced_mode: bool         = field(init=False)
    num_meshes: int                 = field(init=False)
    num_cameras: int                = field(init=False)
    vertices: torch.Tensor          = field(init=False)
    faces: torch.Tensor             = field(init=False)
    ranges: torch.Tensor            = field(init=False)
    v_clipspace: torch.Tensor       = field(init=False)
    view_matrix: torch.Tensor       = field(init=False)
    projection_matrix: torch.Tensor = field(init=False)

def expand_per_camera_tensor(pre_rast: PreRasterizationData, tensor: torch.Tensor):
    # TODO: Handle ranged mode?
    return tensor.repeat((pre_rast.num_meshes, *([1]*(len(tensor.shape)-1))))

def expand_per_mesh_tensor(pre_rast: PreRasterizationData, tensor: torch.Tensor):
    # TODO: Handle ranged mode?
    return tensor.repeat_interleave(pre_rast.num_cameras, 0)

def prepare_batched_rasterization(vertices: torch.Tensor, faces: torch.Tensor, ranges: torch.Tensor, view_matrix: torch.Tensor, projection_matrix: torch.Tensor):
    device = vertices.device

    pre_rast = PreRasterizationData()    
    pre_rast.is_instanced_mode = len(vertices.shape) == 3

    # Determine mesh and camera batch size and alias them
    pre_rast.num_meshes  = vertices.shape[0] if pre_rast.is_instanced_mode else ranges.shape[0]
    pre_rast.num_cameras = view_matrix.shape[0] 
    B = pre_rast.num_meshes
    C = pre_rast.num_cameras

    # Store vertices in homogeneous coordinates and the camera matrices
    pre_rast.vertices    = homogeneous(vertices)
    pre_rast.faces       = faces
    pre_rast.ranges      = ranges
    pre_rast.view_matrix       = view_matrix
    pre_rast.projection_matrix = projection_matrix

    # Handle meshes and cameras as a single batch of size B*L
    # The data structures meshes = [M1, M2] and cameras = [C1, C2]
    # are repeated/expanded to meshes = [M1, M1, M2, M2] and cameras = [C1, C2, C1, C2]
    if pre_rast.is_instanced_mode:
        pre_rast.view_matrix       = pre_rast.view_matrix.repeat((B, 1, 1))
        pre_rast.projection_matrix = pre_rast.projection_matrix.repeat((B, 1, 1))
        pre_rast.vertices          = pre_rast.vertices.repeat_interleave(repeats=C, dim=0)
    else:
        # TODO: Implement and fix ranged mode
        num_vertices      = pre_rast.vertices.shape[0]
        offsets           = num_vertices * torch.arange(C, device=device) 
        pre_rast.ranges   = pre_rast.ranges[None].expand((C, -1, -1)) + offsets[:, 1, 1] # Shape (L,B,2)
        pre_rast.vertices = pre_rast.vertices[None].expand((C, -1, -1))         # Shape (L,V,3)

        pre_rast.ranges = pre_rast.ranges.reshape(-1, 2) # Shape (L*B,2)

    # Transform vertices to clip space
    # TODO: Could be precomputed
    mvp = torch.bmm(pre_rast.projection_matrix, pre_rast.view_matrix)

    if pre_rast.is_instanced_mode:
        # mvp                   = (B*L,4,4)
        # homogeneous(vertices) = (B*L,V,4)
        pre_rast.v_clipspace = torch.bmm(pre_rast.vertices, mvp.transpose(-1, -2))
    else:
        # mvp                   = (L,4,4)
        # homogeneous(vertices) = (L,4)
        # TODO: Implement
        raise NotImplementedError("Range mode not yet implemented.")
    
    # Ensure contiguous tensors
    pre_rast.v_clipspace = pre_rast.v_clipspace.contiguous()
    pre_rast.faces       = pre_rast.faces.contiguous()
    pre_rast.ranges      = pre_rast.ranges.contiguous() if pre_rast.ranges is not None else pre_rast.ranges

    return pre_rast