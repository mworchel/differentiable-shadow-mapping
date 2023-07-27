import copy
from dataclasses import dataclass
from enum import Enum
import nvdiffrast.torch as dr 
import torch
from typing import List, Tuple, Union
import warnings

from .common import repeat_dim
from .filtering import apply_filter_2d
from .light import DirectionalLights, SpotLights, PointLights, UnidirectionalLights
from .rasterization import PreRasterizationData, prepare_batched_rasterization, expand_per_camera_tensor
from .transformation import homogeneous, hnormalized

def chebyshev_one_sided(variance, mean, x):
    return variance / (variance + (x - mean)**2)

class ShadowMethod(Enum):
    Standard           = 0
    VarianceShadowMaps = 1

@dataclass
class ShadowMapBase:
    light: Union[DirectionalLights, SpotLights, PointLights]
    pre_rast: PreRasterizationData
    rast_out: torch.Tensor

@dataclass
class StandardShadowMap(ShadowMapBase):
    depth: torch.Tensor

@dataclass
class VarianceShadowMap(ShadowMapBase):
    m1: torch.Tensor
    m2: torch.Tensor

def render_shadow_map(context: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext], 
                      vertices: torch.Tensor, faces: torch.Tensor, ranges: torch.Tensor, 
                      light: Union[DirectionalLights, SpotLights, PointLights],
                      method: ShadowMethod = ShadowMethod.VarianceShadowMaps) -> Union[StandardShadowMap, VarianceShadowMap]:
    """ Render shadow maps for an array of meshes using an array of light sources. For additional info on parameters, see `nvdiffrast.rasterize`.
    
    Args:
        context: The nvdiffrast rendering context
        vertices: Set of mesh vertices; if batched then instanced mode is used ((B,)V,3)
        faces: Indices of the indexed face set                                 (F,3)
        ranges: Ranges (disabled in instanced mode)                            (B,2)
        light: The batch of light sources
        method: Shadow mapping method determining the type of shadow map to create

    Returns:
        A single struct representing an array of shadow maps
    """

    # Perform batched rasterization
    pre_rast              = prepare_batched_rasterization(vertices, faces, ranges, light.view_matrix, light.projection_matrix)
    rast_out, rast_out_db = dr.rasterize(context, pos=pre_rast.v_clipspace, tri=pre_rast.faces, resolution=light.resolution, ranges=pre_rast.ranges, grad_db=False)

    # Compute view-space vertex positions (note that pre_rast has vertices in homogeneous coordinates)
    v_viewspace = torch.bmm(pre_rast.vertices, pre_rast.view_matrix.transpose(-1, -2))
    v_viewspace = hnormalized(v_viewspace)

    # Compute the view-space depth
    position_mv, _ = dr.interpolate(v_viewspace.contiguous(), rast_out, pre_rast.faces, rast_db=rast_out_db, diff_attrs=None) 
    depth          = torch.norm(position_mv, dim=-1, keepdim=True)

    # Invalidate out-of-mask depths
    # Note: `light.far` is *per light* but point lights have six views per light
    # TODO: Make this optional?
    mask  = rast_out[:, :, :, 3:4] > 0
    far   = light.far if not isinstance(light, PointLights) else light.far.repeat_interleave(6, 0)
    far   = expand_per_camera_tensor(pre_rast, far)[:, None, None, None]
    depth = torch.where(mask, depth, far)

    if method == ShadowMethod.Standard:
        return StandardShadowMap(light, pre_rast, rast_out, depth)
    elif method == ShadowMethod.VarianceShadowMaps:
        m1 = depth
        m2 = depth*depth
        return VarianceShadowMap(light, pre_rast, rast_out, m1, m2)
    else:
        raise RuntimeError(f"Unknown shadow mapping method '{str(method)}'.")

def filter_shadow_map(shadow_map: Union[StandardShadowMap, VarianceShadowMap], antialias: bool=True, kernel: torch.Tensor=None) -> Union[StandardShadowMap, VarianceShadowMap]:
    """ Filter an array of shadow maps
    
    Args:
        antialias: Indicator if nvdiffrast's antialiasing is applied to the shadow map
        kernel: Kernel for explicit filtering, with shape (H,W)

    Returns:
        The filtered shadow maps
    """

    if isinstance(shadow_map, StandardShadowMap):
        warnings.warn("Standard shadow maps are not designed to be filtered.")

    if isinstance(shadow_map.light, PointLights):
        warnings.warn("Omni-directional point light support is preliminary: filtering can generate visible seams.")

    pre_rast = shadow_map.pre_rast
    rast_out = shadow_map.rast_out

    # Do not modify the passed struct in-place
    shadow_map = copy.copy(shadow_map)

    # Anti-alias the shadow map
    if antialias:
        if isinstance(shadow_map, StandardShadowMap):
            shadow_map.depth = dr.antialias(shadow_map.depth, rast_out, pre_rast.v_clipspace, pre_rast.faces)
        elif isinstance(shadow_map, VarianceShadowMap):
            shadow_map.m1 = dr.antialias(shadow_map.m1, rast_out, pre_rast.v_clipspace, pre_rast.faces)
            shadow_map.m2 = dr.antialias(shadow_map.m2, rast_out, pre_rast.v_clipspace, pre_rast.faces)

    # Explicitly filter the shadow map
    if kernel is not None:
        kernel = kernel[None, None, :, :] if len(kernel.shape) == 2 else kernel # Expand channel dimensions

        if isinstance(shadow_map, StandardShadowMap):
            shadow_map.depth = apply_filter_2d(shadow_map.depth, kernel, padding_mode='replicate')
        elif isinstance(shadow_map, VarianceShadowMap):
            shadow_map.m1 = apply_filter_2d(shadow_map.m1, kernel, padding_mode='replicate') #, mode='constant', value=light.far)
            shadow_map.m2 = apply_filter_2d(shadow_map.m2, kernel, padding_mode='replicate') #, mode='constant', value=light.far**2)

    return shadow_map

def map_points_to_light_space(light: Union[DirectionalLights, SpotLights, PointLights], points: torch.tensor):
    """ Map an array of points into light space.

    Args:
        light: A set of light sources defined by L cameras
        points: 3D points with shape (B,...,3)

    Returns:
        light_to_point: Vector from the light source to the points with shape (B*L,...,3)
                        For unidirectional lights, this vector is in view space, for point lights, it is in world space
        light_direction: Incident light direction in world space (from points to light) with shape (B*L,...,3)
        uv: uv coordinates for sampling the light source image space with shape (B*L,...,2) for unidirectional lights 
            and (B*L//6,...,3) for omnidirectional point lights.
    """

    B = points.shape[0]
    M = points.shape[1:-1]
    L = light.view_matrix.shape[0]

    if isinstance(light, UnidirectionalLights):
        # Align shapes of positions and view matrices
        # Given meshes = [M1, M2], cameras = [C1, C2] and lights = [L1, L2],
        # the data structures are expanded to meshes = [M1, M1, M2, M2], lights = [L1, L2, L1, L2]
        points_homogeneous = homogeneous(points).reshape(B, -1, 4).repeat_interleave(L, 0) # (B,...,4) -> (B,M,4) -> (B*L,M,4)
        view_matrix        = light.view_matrix.repeat((B, 1, 1))                           # (L,4,4) -> (B*L,4,4)
        projection_matrix  = light.projection_matrix.repeat((B, 1, 1))                     # (L,4,4) -> (B*L,4,4)

        # Compute positions in light view-space
        points_viewspace = torch.bmm(points_homogeneous, view_matrix.transpose(-1, -2))
        uv               = hnormalized(torch.bmm(points_viewspace, projection_matrix.transpose(-1, -2)))
        uv               = uv[:, :, :2].reshape(-1, *M, 2)

        # Compute light to point vectors (in view space)
        light_to_point = hnormalized(points_viewspace)

        if isinstance(light, DirectionalLights):
            # For directional lights, -light_to_point is not the actual direction to the light
            # but it is given by the direction stored with the lights
            light_direction = repeat_dim(light.direction, B, 0)
            light_direction = light_direction.view(B*L, *[1]*len(M), 3).expand(B*L, *M, 3)
        else:
            # Invert the rotational part of the view matrix to get vectors in world space
            # NOTE: Assumes view_matrix[:3, :3] is a rotation matrix
            light_to_point_world = torch.bmm(light_to_point, view_matrix[:, :3, :3])
            light_direction      = -torch.nn.functional.normalize(light_to_point_world, p=2, dim=-1)
        light_direction = light_direction.reshape(-1, *M, 3)

        light_to_point = light_to_point.reshape(-1, *M, 3)

        return light_to_point, light_direction, uv
    else: # PointLights
        light_to_point = points[:, None] - light.position.view(1, -1, *[1]*len(M), 3) # (L//6,3) -> (B,L//6,...,3) 
        light_to_point = light_to_point.reshape(B*L//6, *M, 3)                        # ... -> (B*L//6,...,3)
        uv             = torch.nn.functional.normalize(light_to_point, p=2, dim=-1)

        # Light direction is from point to light, so flip the sign
        light_direction = -uv

        return light_to_point, light_direction, uv

def sample_light_space_image(light: Union[UnidirectionalLights, PointLights], image: torch.Tensor, uv: torch.Tensor, filter_mode: str, boundary_mode: str):
    """ Sample an image defined in light space

    Args: 
        light: A set of light sources, represented by L view matrices
        image: A set of images with shape (B*L,H,W,C)
        uv: uv coordinates with shape (B*L,H,W,2) for unidirectional lights and (B*L//6,H,W,3) for point lights.
        filter_mode: Filter mode used for interpolation, see nvdiffrast.texture for details
        boundary_mode: Boundary mode used for out-of-texture accesses, see nvdiffrast.texture for details (ignored for point lights)
    """

    if isinstance(light, UnidirectionalLights):
        # FIXME: Filter mode 'linear' with boundary mode 'zero' is broken in nvdiffrast < 0.3.1
        # Transform uv coordinates to nvdiffrast format for usage with `dr.texture`
        # uv     = 0.5*(uv[:, :, :2]+1)
        # return = dr.texture(image, uv, filter_mode=filter_mode, boundary_mode=boundary_mode)
        filter_mode_map = {
            'nearest': 'nearest',
            'linear': 'bilinear'
        }
        boundary_mode_map = {
            'zero': 'zeros',
            'clamp': 'border',
        }
        return torch.nn.functional.grid_sample(image.permute(0, 3, 1, 2), uv, mode=filter_mode_map[filter_mode], padding_mode=boundary_mode_map[boundary_mode]).permute(0, 2, 3, 1)
    else: # PointLights
        BL,H,W,C = image.shape
        image    = image.reshape(BL//6, 6, H, W, C)
        return dr.texture(image, uv=uv, filter_mode=filter_mode, boundary_mode='cube')

def compute_visibility(shadow_map: Union[StandardShadowMap, VarianceShadowMap], positions: torch.Tensor, **kwargs):
    """ Compute the visibility for a set of points for an array light sources using their corresponding shadow maps.

    Args:
        positions: World-space position map to compute visibility for with shape (B,C,H,W,3)
                   B is the number of meshes/scenes and C is the number of cameras. 
        shadow_map: The shadow map

    Returns:
        A visibility map with shape (B,C,L,H,W,1) where L is the number of light sources.
    """
    assert len(positions.shape) == 5
    assert positions.shape[-1] == 3

    light = shadow_map.light

    B, C, H, W = positions.shape[0:4]
    L          = light.view_matrix.shape[0]

    # Compute the depth as distance to the light source
    light_to_point, _, uv = map_points_to_light_space(shadow_map.light, positions.reshape(B*C, H, W, 3))
    depth_actual          = torch.norm(light_to_point, dim=-1, keepdim=True)

    # Shadow map tensors have shape (B*L,H,W,1) and need to be expanded to (B*C*L,H,W,1)
    def expand_shadow_tensor(tensor: torch.Tensor):
        _, H, W, D = tensor.shape
        return tensor.reshape(B, 1, L, H, W, D).expand(B, C, L, H, W, D).reshape(B*C*L, H, W, D)

    visibility = None
    if isinstance(shadow_map, StandardShadowMap):
        depth_sampled = sample_light_space_image(shadow_map.light, expand_shadow_tensor(shadow_map.depth), uv, filter_mode='nearest', boundary_mode='zero')

        # Standard depth test (with bias)
        bias       = kwargs.get('bias', 0)
        visibility = (depth_actual <= depth_sampled + bias).to(dtype=torch.float32)
    elif isinstance(shadow_map, VarianceShadowMap):
        m1_sampled = sample_light_space_image(shadow_map.light, expand_shadow_tensor(shadow_map.m1), uv=uv, filter_mode='linear', boundary_mode='zero')
        m2_sampled = sample_light_space_image(shadow_map.light, expand_shadow_tensor(shadow_map.m2), uv=uv, filter_mode='linear', boundary_mode='zero')

        mean_sampled     = m1_sampled
        variance_sampled = m2_sampled - m1_sampled*m1_sampled

        # Variance computation is numerically instable, so clamp it to a small positive value
        variance_min     = kwargs.get('variance_min', 0.0001)
        variance_sampled = variance_sampled.clamp(min=variance_min)

        # Visibility function of Variance Shadow Mapping based on the one-sided Chebyshev inequality
        visibility = torch.where(depth_actual <= mean_sampled, 
            torch.tensor(1, dtype=torch.float32, device=positions.device), 
            chebyshev_one_sided(variance_sampled, mean_sampled, depth_actual)
        ).clamp(0, 1)
    else:
        raise NotImplementedError(f"Shadow map type '{type(shadow_map)}' has no visibility computation.")

    # For unidirectional lights, discard points behind the light source
    if isinstance(shadow_map.light, UnidirectionalLights):
        visibility = torch.where(light_to_point[:, :, :, 2:3] <= 0, visibility, 0)

    # For point lights, L is not the number of lights but L/6 (each cubemap corresponds to six views)
    num_lights = L if isinstance(shadow_map.light, UnidirectionalLights) else L//6
    return visibility.reshape(B, C, num_lights, H, W, -1)