import torch
from typing import List

from .scene import Mesh

def construct_frame(normal):
    normal = torch.nn.functional.normalize(normal, p=2, dim=0)

    sin_theta = normal[1]

    # Handle degenerate cases
    if sin_theta == 1:
        tangent = torch.tensor([0, 0, -1], dtype=torch.float32, device=normal.device)
        bitangent = torch.tensor([-1, 0, 0], dtype=torch.float32, device=normal.device)
    elif sin_theta == -1:
        tangent = torch.tensor([0, 0, 1], dtype=torch.float32, device=normal.device)
        bitangent = torch.tensor([-1, 0, 0], dtype=torch.float32, device=normal.device)
    else: 
        phi = torch.atan2(normal[0], normal[2])
        theta = torch.asin(sin_theta)

        # dn/dphi
        tangent = torch.stack([
            torch.cos(phi)*torch.cos(theta),
            torch.zeros_like(sin_theta),
            -torch.sin(phi)*torch.cos(theta)
        ])

        # dn/dtheta
        bitangent = torch.stack([
            -torch.sin(phi)*sin_theta,
            torch.cos(theta),
            -torch.cos(phi)*sin_theta
        ])

    tangent = torch.nn.functional.normalize(tangent, p=2, dim=0)
    bitangent = torch.nn.functional.normalize(bitangent, p=2, dim=0)

    return normal, tangent, bitangent

def azimuth_elevation_to_direction(azimuth, elevation):
    if not torch.is_tensor(azimuth):
        azimuth = torch.tensor(azimuth, dtype=torch.float32)

    if not torch.is_tensor(elevation):
        elevation = torch.tensor(elevation, dtype=torch.float32, device=azimuth.device)

    # Elevation is measured from the horizon, the azimuth
    # from the x-axis, and the y-axis points up
    return torch.stack([
        torch.cos(elevation) * torch.cos(azimuth), 
        torch.sin(elevation),
        -torch.cos(elevation) * torch.sin(azimuth),
    ], dim=-1)

def create_plane_mesh(position, normal, size, device):
    v = torch.tensor([[-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0]], dtype=torch.float32, device=device)
    f = torch.tensor([[0, 1, 2], [2, 1, 3]], dtype=torch.int32, device=device)

    normal = normal if torch.is_tensor(normal) else torch.tensor(normal, dtype=torch.float32, device=device)
    position = position if torch.is_tensor(position) else torch.tensor(position, dtype=torch.float32, device=device)

    normal = torch.nn.functional.normalize(normal, p=2, dim=0)
    n, t, b = construct_frame(normal)

    R = torch.stack([t, b, n], dim=1)

    return Mesh((size*v @ R.T) + position[None, :], f, n[None, :].repeat(4, 1))

def find_edges(indices, remove_duplicates=True):
    # Extract the three edges (in terms of vertex indices) for each face 
    # edges_0 = [f0_e0, ..., fN_e0]
    # edges_1 = [f0_e1, ..., fN_e1]
    # edges_2 = [f0_e2, ..., fN_e2]
    edges_0 = torch.index_select(indices, 1, torch.tensor([0,1], device=indices.device))
    edges_1 = torch.index_select(indices, 1, torch.tensor([1,2], device=indices.device))
    edges_2 = torch.index_select(indices, 1, torch.tensor([2,0], device=indices.device))

    # Merge the into one tensor so that the three edges of one face appear sequentially
    # edges = [f0_e0, f0_e1, f0_e2, ..., fN_e0, fN_e1, fN_e2]
    edges = torch.cat([edges_0, edges_1, edges_2], dim=1).view(indices.shape[0] * 3, -1)

    if remove_duplicates:
        edges, _ = torch.sort(edges, dim=1)
        edges = torch.unique(edges, dim=0)

    return edges

def find_connected_faces(indices):
    edges = find_edges(indices, remove_duplicates=False)

    # Make sure that two edges that share the same vertices have the vertex ids appear in the same order
    edges, _ = torch.sort(edges, dim=1)

    # Now find edges that share the same vertices and make sure there are only manifold edges
    _, inverse_indices, counts = torch.unique(edges, dim=0, sorted=False, return_inverse=True, return_counts=True)
    assert counts.max() == 2

    # We now create a tensor that contains corresponding faces.
    # If the faces with ids fi and fj share the same edge, the tensor contains them as
    # [..., [fi, fj], ...]
    face_ids = torch.arange(indices.shape[0])               
    face_ids = torch.repeat_interleave(face_ids, 3, dim=0) # Tensor with the face id for each edge

    face_correspondences = torch.zeros((counts.shape[0], 2), dtype=torch.int64)
    face_correspondences_indices = torch.zeros(counts.shape[0], dtype=torch.int64)

    # ei = edge index
    for ei, ei_unique in enumerate(list(inverse_indices.cpu().numpy())):
        face_correspondences[ei_unique, face_correspondences_indices[ei_unique]] = face_ids[ei] 
        face_correspondences_indices[ei_unique] += 1

    return face_correspondences.to(device=indices.device)[counts == 2]

def merge_meshes(meshes: List[Mesh]):
    assert(len(meshes) > 0)

    v = []
    f = []
    n = []
    d = []
    num_vertices = 0
    for mesh in meshes:
        # TODO: Apply model transform here
        v += [ mesh.vertices ]
        f += [ mesh.faces + num_vertices ]
        n += [ mesh.normals ]
        # OK: Diffuse albedo is tricky
        diffuse_albedo = mesh.diffuse_albedo
        if diffuse_albedo is None:
            # FIXME: If any mesh has RGB diffuse albedo, this line will cause an error
            diffuse_albedo = torch.tensor([[1]], device=mesh.vertices.device, dtype=torch.float32).expand(mesh.vertices.shape[0], -1)
        elif len(diffuse_albedo.shape) == 1:
            diffuse_albedo = diffuse_albedo[None, :].expand(mesh.vertices.shape[0], -1)
        d += [ diffuse_albedo ]
        num_vertices += mesh.vertices.shape[0]
    v = torch.cat(v, dim=0)
    f = torch.cat(f, dim=0)
    n = torch.cat(n, dim=0)
    d = torch.cat(d, dim=0)

    return Mesh(v, f, n, d=d)

def weld_seams(mesh: Mesh):
    # Algorithms for uv mapping sometimes duplicate vertices, so we have vertices with the same position but different uv coordinates.
    # When deforming the mesh, we want to treat these duplicates like they are the same vertex.
    vertices_unique, vertices_inverse_indices = torch.unique(mesh.vertices, sorted=False, return_inverse=True, dim=0)

    # Remap indices to the unique vertex ids so we can e.g. detect connected faces across uv borders
    indices_unique = vertices_inverse_indices[mesh.faces.to(dtype=torch.long)]

    return Mesh(vertices_unique, indices_unique.to(dtype=torch.int32), n=None)