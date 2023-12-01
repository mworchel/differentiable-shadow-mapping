from enum import Enum
import math
import nvdiffrast.torch as dr
import torch
from typing import List, Optional, Union, Tuple

from .geometry import merge_meshes
from .scene import Mesh, DirectionalLight, SpotLight, PointLight, Camera

from diffshadow.common import to_tensor, dot
from diffshadow.filtering import get_box_filter_2d, get_gaussian_filter_2d, get_wendland_filter_2d
from diffshadow.transformation import hnormalized, homogeneous, apply_transformation
from diffshadow.light import DirectionalLights, SpotLights, PointLights
from diffshadow.shadow import render_shadow_map, filter_shadow_map, map_points_to_light_space, sample_light_space_image, compute_visibility, ShadowMethod, VarianceShadowMap

class KernelType(Enum):
    Box      = 0
    Gaussian = 1
    Wendland = 2

class SimpleRenderer:
    """ A simple renderer that uses the shadow mapping primitives
    """

    def __init__(self, context: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]) -> None:
        self.context = context

    def _create_core_light_source(self, light: Union[DirectionalLight, SpotLight, PointLight], device: torch.device) -> Union[DirectionalLights, SpotLights, PointLights]:
        if isinstance(light, DirectionalLight):
            return DirectionalLights.create(light.direction, light.distance, light.size, light.near, light.far, light.resolution, device=device)
        elif isinstance(light, SpotLight):
            return SpotLights.create(light.view_matrix, light.fovy, light.near, light.far, light.resolution, device=device)
        elif isinstance(light, PointLight):
            return PointLights.create(light.position, light.near, light.far, light.resolution, device=device)

    def _map_to_light_ndc(self, points: torch.Tensor, light: Union[DirectionalLights, SpotLights]):
        mvp    = light.projection_matrix[0] @ light.view_matrix[0]
        xyzw   = homogeneous(points) @ mvp.T
        w      = xyzw[:, :, 3:4]
        xyz    = hnormalized(xyzw)
        return xyz, w
    
    def _get_light_intensity(self, light: Union[DirectionalLight, SpotLight, PointLight], light_: Union[DirectionalLights, SpotLights, PointLights], uv: torch.Tensor):
        if light.intensity is None:
            return 1.0

        intensity = to_tensor(light.intensity, dtype=torch.float32, device=uv.device)

        # Expand to image shape and exit if intensity is constant
        if len(intensity.shape) == 1:
            return intensity[None, None]
        
        # Sample intensity in light space
        if len(intensity.shape) == 3: 
            intensity = intensity[None]
        # else: has shape [6,H,W,C], for point lights

        return sample_light_space_image(light_, intensity, uv, filter_mode='linear', boundary_mode='zero')[0]

    def _shadow_pass(self, mesh: Mesh, lights_: List[Union[DirectionalLights, SpotLights, PointLights]], smoothing_kernel_width: int, smoothing_kernel: KernelType, antialias=True) -> List[VarianceShadowMap]:
        device = mesh.vertices.device

        # Compute shadow maps by looping over the light sources
        # (this is inefficient if there are many lights; can be improved by batching similar lights)
        shadow_maps: List[VarianceShadowMap] = []
        for light_ in lights_:
            shadow_maps += [ render_shadow_map(self.context, vertices=mesh.vertices[None], faces=mesh.faces, ranges=None, light=light_, method=ShadowMethod.VarianceShadowMaps) ]

        # Filter shadow maps
        kernel: Optional[torch.Tensor] = None
        if smoothing_kernel_width > 0:
            if smoothing_kernel == KernelType.Box:
                kernel = get_box_filter_2d(smoothing_kernel_width).to(device)
            elif smoothing_kernel == KernelType.Gaussian:
                kernel = get_gaussian_filter_2d(smoothing_kernel_width).to(device)
            elif smoothing_kernel == KernelType.Wendland:
                kernel = get_wendland_filter_2d(smoothing_kernel_width).to(device)
            else:
                raise RuntimeError(f"Unknown smoothing kernel {smoothing_kernel}")

        for i in range(len(shadow_maps)):
            shadow_maps[i] = filter_shadow_map(shadow_maps[i], antialias=antialias, kernel=kernel)

        return shadow_maps

    def _gbuffer_pass(self, mesh: Mesh, camera: Camera, resolution: Tuple[int, int]):
        v_clipspace = apply_transformation(mesh.vertices, camera.projection_matrix @ camera.view_matrix)
        rast_out, _ = dr.rasterize(self.context, v_clipspace[None], mesh.faces, resolution, grad_db=False)

        gbuffer = {
            'rast_out': rast_out,
            'v_clipspace': v_clipspace,
            'f': mesh.faces
        }

        # World-space positions
        positions, _ = dr.interpolate(mesh.vertices[None], rast_out, mesh.faces)
        gbuffer['position'] = positions[0]

        # World-space normals
        n_worldspace = (mesh.normals @ mesh.transform_inv_transposed.T[:3, :3])
        normals, _ = dr.interpolate(n_worldspace[None], rast_out, mesh.faces)
        gbuffer['normal'] = normals[0]
        gbuffer['normal'] = torch.nn.functional.normalize(gbuffer['normal'], p=2, dim=-1)

        gbuffer['mask'] = rast_out[0, :, :, 3] > 0

        # Diffuse albedo
        diffuse_albedo = mesh.diffuse_albedo
        if diffuse_albedo is None:
            diffuse_albedo = torch.tensor([[1]], device=mesh.vertices.device, dtype=torch.float32).expand(mesh.vertices.shape[0], -1)
        elif len(diffuse_albedo.shape) == 1:
            diffuse_albedo = diffuse_albedo[None, :].expand(mesh.vertices.shape[0], -1)
        albedo, _ = dr.interpolate(diffuse_albedo[None], rast_out, mesh.faces)
        gbuffer['diffuse_albedo'] = albedo[0]

        return gbuffer


    def render(self, meshes: List[Mesh], lights: List[Union[DirectionalLight, SpotLight, PointLight]], camera: Camera, resolution: Tuple[int, int]=(512, 512), ambient: float = 0.1, background: float = 0, use_shadows: bool = True, smoothing_kernel_width: int = 3, smoothing_kernel: KernelType = KernelType.Box, use_shadow_antialiasing: bool = True, return_visibility: bool = False, return_mask: bool = False, return_gbuffer: bool = False):
        # Apply model matrices and merge all meshes
        scene_mesh = merge_meshes([m.with_applied_transform() for m in meshes])

        # Convert each light instance into a batched light instance compatible with the shadow primitives
        lights_ = [ self._create_core_light_source(light, scene_mesh.vertices.device) for light in lights ]

        # The shadow pass renders shadow maps for all active light sources
        shadow_maps = self._shadow_pass(scene_mesh, lights_, smoothing_kernel_width=smoothing_kernel_width, smoothing_kernel=smoothing_kernel, antialias=use_shadow_antialiasing) if use_shadows else []

        # The G-buffer pass obtains data for the camera view (positions, normals, diffuse albedo, ...)
        gbuffer = self._gbuffer_pass(scene_mesh, camera, resolution)

        color        = gbuffer['diffuse_albedo'] * ambient
        visibilities = []
        for i, light in enumerate(lights):
            light_ = lights_[i]

            light_to_point, L, uv = map_points_to_light_space(light_, gbuffer['position'][None])
            light_to_point = light_to_point[0]
            L              = L[0]

            # Point lights have radial fall-off 
            falloff   = 1.0 if not isinstance(light, PointLight) else 1/light_to_point.norm(dim=-1, keepdim=True)**2

            # Intensity is the "color" of the light; can be spatially varying
            intensity = self._get_light_intensity(light, light_, uv)

            NdotL      = dot(gbuffer['normal'], L, keepdims=True)
            irradiance = gbuffer['diffuse_albedo'] / math.pi * intensity * NdotL.clamp(min=0, max=1) * falloff

            if use_shadows:
                visibility = compute_visibility(shadow_maps[i], gbuffer['position'][None, None])[0, 0, 0]
                irradiance = irradiance * visibility
                visibilities += [ visibility ]
            elif isinstance(light, SpotLight):
                # Even if not casting shadows, spot lights only illuminate surfaces
                # in front of them and inside their field-of-view.
                # `compute_visibility` handles this implicitly
                xyz, w  = self._map_to_light_ndc(gbuffer['position'], light_)
                x = xyz[:, :, 0:1]
                y = xyz[:, :, 1:2]
                visibility = torch.where(
                    (w >= 0) & 
                    (x >= -1) & (x <= 1) &
                    (y >= -1) & (y <= 1), 1, 0)
                irradiance = irradiance * visibility

            color += irradiance

        color = torch.where(gbuffer['mask'][:, :, None] > 0, color, background)
        color = dr.antialias(color[None], gbuffer['rast_out'], gbuffer['v_clipspace'], scene_mesh.faces)[0]

        results = [color]

        if return_visibility:
            results += [ visibilities ]

        if return_mask:
            results += [ gbuffer['mask'] ]
        
        if return_gbuffer:
            results += [ gbuffer ]

        return results[0] if len(results) == 1 else results