import torch
from typing import List, Tuple, Union

from .common import expand_to_common_batch_size
from .transformation import create_view_matrix_from_direction, create_orthographic_projection_matrix, create_perspective_projection_matrix, create_cubemap_view_matrix

class BaseLights:
    def __init__(self, near: torch.Tensor, far: torch.Tensor, resolution: Tuple[int, int]) -> None:
        assert len(near.shape) == 1, f"near plane distance must be given per light source (received {near.shape=})"
        assert len(far.shape) == 1, f"far plane distance must be given per light source (received {far.shape=})"
        assert near.shape[0] == far.shape[0], "Batch size of near and far plane must match"

        self.near       = near
        self.far        = far
        self.resolution = resolution

class UnidirectionalLights(BaseLights):
    """ A set of unidirectional light sources, where each uses one planar shadow map
    
    Note: The near and far values should coincide with the values used for the projection matrices

    Args:
        view_matrix: View matrices for the light cameras             (L,4,4)
        projection_matrix: Projection matrices for the light cameras (L,4,4)
        near: Near plane distances                                   (L,)
        far: Far plane distances                                     (L,)
        resolution: Resolution of the associated shadow map          (Height,Width)
    """

    def __init__(self, view_matrix: torch.Tensor, projection_matrix: torch.Tensor, near: torch.Tensor, far: torch.Tensor, resolution: Tuple[int, int]) -> None:
        super().__init__(near, far, resolution)
        
        assert len(view_matrix.shape) == 3
        assert len(projection_matrix.shape) == 3

        self.view_matrix              = view_matrix
        self.projection_matrix        = projection_matrix
        self.__view_projection_matrix = None
    
    @property
    def view_projection_matrix(self):
        if self.__view_projection_matrix is None:
            self.__view_projection_matrix = torch.bmm(self.projection_matrix, self.view_matrix)
        return self.__view_projection_matrix

class DirectionalLights(UnidirectionalLights):
    """ A set of directional light sources

    Args:
        direction: Direction vectors pointing towards the light sources (L,3)
        distance: Distance of the virtual eye points from the origin    (L,)
        size: Size of the light image planes in world units             (L,)
        near: Near plane distances                                      (L,)
        far: Far plane distances                                        (L,)
        resolution: Resolution of the associated shadow maps            (Height,Width)
    """

    def __init__(self, direction: torch.Tensor, distance: torch.Tensor, size: torch.Tensor, near: torch.Tensor, far: torch.Tensor, resolution: Tuple[int, int], device: torch.device = None) -> None:
        super().__init__(create_view_matrix_from_direction(-direction, distance), create_orthographic_projection_matrix(size=size, near=near, far=far, device=direction.device), near, far, resolution)
        self.direction = direction  
        self.distance  = distance
        self.size      = size

    @classmethod
    def create(cls, direction: Union[List, torch.Tensor], distance: Union[float, torch.Tensor], size: Union[float, torch.Tensor], near: Union[float, torch.Tensor], far: Union[float, torch.Tensor], resolution: Tuple[int, int], device: torch.device = None):
        """ Convenience interface for creating a single or multiple directional lights from potential non-tensor parameters
        """
        direction, distance, size, near, far, _ = expand_to_common_batch_size(direction, distance, size, near, far, batched_lengths=[2, 1, 1, 1, 1], dtype=torch.float32, device=device)
        return cls(direction, distance, size, near, far, resolution)

class SpotLights(UnidirectionalLights):
    """ A set of spot light sources

    Args:
        view_matrix: View matrices for the spot light cameras             (L,4,4)
        projection_matrix: Projection matrices for the spot light cameras (L,4,4)
        near: Near plane distances                                        (L,)
        far: Far plane distances                                          (L,)
        resolution: Resolution of the associated shadow map               (Height,Width)
    """

    def __init__(self, view_matrix: torch.Tensor, fovy: torch.Tensor, near: torch.Tensor, far: torch.Tensor, resolution: Tuple[int, int]) -> None:
        super().__init__(view_matrix, create_perspective_projection_matrix(fovy, near=near, far=far, device=view_matrix.device), near, far, resolution)
        self.fovy = fovy

    @classmethod
    def create(cls, view_matrix: torch.Tensor, fovy: Union[float, torch.Tensor], near: Union[float, torch.Tensor], far: Union[float, torch.Tensor], resolution: Tuple[int, int], device: torch.device = None):
        """ Convenience interface for creating a single or multiple spot lights from potential non-tensor parameters
        """
        view_matrix, fovy, near, far, _ = expand_to_common_batch_size(view_matrix, fovy, near, far, batched_lengths=[3, 1, 1, 1], dtype=torch.float32, device=device)
        return cls(view_matrix, fovy, near, far, resolution)

class PointLights(BaseLights):
    """ A set of omnidirectional point light sources

    Args:
        position: Positions of the light sources            (L,4,4)
        near: Near plane distances                          (L,)
        far: Far plane distances                            (L,)
        resolution: Resolution of the associated shadow map (Height,Width)
    """

    def __init__(self, position: torch.Tensor, near: Union[float, torch.Tensor], far: Union[float, torch.Tensor], resolution: Tuple[int, int]):
        super().__init__(near, far, resolution)
        self.position = position

        # Create view and projection matrices for shadow map rendering
        # (L,6,4,4) --reshape--> (L*6,4,4)
        self.view_matrix       = torch.stack([create_cubemap_view_matrix(position, i) for i in range(6)], dim=1).reshape(-1, 4, 4)
        self.projection_matrix = create_perspective_projection_matrix(fovy=90, aspect=resolution[1]/resolution[0], near=near, far=far, device=position.device).repeat_interleave(6, 0)

    @classmethod
    def create(cls, position: Union[List[float], torch.Tensor], near: Union[float, torch.Tensor], far: Union[float, torch.Tensor], resolution: Tuple[int, int], device: torch.device = None):
        """ Convenience interface for creating a single or multiple point lights from potential non-tensor parameters
        """
        position, near, far, _ = expand_to_common_batch_size(position, near, far, batched_lengths=[2, 1, 1], dtype=torch.float32, device=device)
        return cls(position, near, far, resolution)