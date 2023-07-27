import torch

from diffshadow.transformation import apply_transformation

class Mesh:
    def __init__(self, v, f, n=None, d=None, transform=torch.eye(4)):
        self.vertices = v
        self.faces = f
        self.diffuse_albedo = d
        self.transform = transform
        if n is None:
            self.compute_normals()
        else:
            self.normals = n
    
    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform_new):
        self._transform = transform_new.to(self.vertices.device)
        self._transform_inv_transposed = torch.inverse(self._transform).T

    @property
    def transform_inv_transposed(self):
        return self._transform_inv_transposed

    def with_applied_transform(self):
        vertices = apply_transformation(self.vertices, self.transform, mode='affine')
        normals  = apply_transformation(self.normals, self.transform_inv_transposed, mode='euclidean')
        return self.with_vertices(vertices, normals)

    def with_vertices(self, vertices, normals=None):
        return Mesh(v=vertices, f=self.faces, n=normals, d=self.diffuse_albedo)

    def compute_normals(self):
        indices = self.faces.to(dtype=torch.long)

        # Compute the face normals
        a = self.vertices[indices][:, 0, :]
        b = self.vertices[indices][:, 1, :]
        c = self.vertices[indices][:, 2, :]
        #self.face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1) 
        self.face_normals = torch.cross(b - a, c - a)

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(self.vertices)
        vertex_normals = vertex_normals.index_add(0, indices[:, 0], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, indices[:, 1], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, indices[:, 2], self.face_normals)
        self.normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1) 

class Camera:
    def __init__(self, view_matrix, projection_matrix):
        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix

class DirectionalLight:
    def __init__(self, direction, intensity=None, near=0.1, far=10, size=2, resolution=(256, 256), distance=5):
        self.direction = direction
        self.intensity = intensity
        self.near = near
        self.far = far
        self.size = size
        self.resolution = resolution
        self.distance = distance

class SpotLight:
    def __init__(self, view_matrix, fovy, intensity=None, near=0.1, far=10, resolution=(256, 256)):
        self.view_matrix = view_matrix
        self.fovy = fovy
        self.intensity = intensity
        self.near = near
        self.far = far
        self.resolution = resolution

class PointLight:
    def __init__(self, position, range=None, intensity=None, near=0.01, far=10, resolution=(256, 256)):
        self.position = position
        self.range = range if range is not None else far
        self.intensity = intensity
        self.near = near
        self.far = far
        self.resolution = resolution