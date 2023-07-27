import numpy as np
import trimesh
import torch

def load_mesh(path, device, normalize=True, with_vertex_color=False):
    mesh: trimesh.Trimesh = trimesh.load_mesh(path)
    v = mesh.vertices
    if normalize:
        v -= np.mean(v, axis=0)
        v /= np.max(np.linalg.norm(v, axis=-1))
    f = mesh.faces

    v = torch.from_numpy(v).to(device, dtype=torch.float32)
    f = torch.from_numpy(f).to(device, dtype=torch.int64)

    vertex_normals = torch.from_numpy(mesh.vertex_normals).to(device, dtype=torch.float32)

    result = [v, f.to(dtype=torch.int32), vertex_normals]

    if with_vertex_color:
        result += [ torch.from_numpy(mesh.visual.vertex_colors).to(device, dtype=torch.float32)[..., :3] / 255.0 ]

    return tuple(result)

def load_quad_mesh(path, device):
    import openmesh as om

    mesh = om.read_polymesh(path, halfedge_tex_coord = True)

    # Convert the quad mesh to a triangle mesh by building an indexed face set
    v = mesh.points()
    f_quad = mesh.fv_indices()
    f = np.zeros((2*f_quad.shape[0], 3), dtype=np.int32)
    f[0::2, :] = f_quad[:, [0, 1, 2]]
    f[1::2, :] = f_quad[:, [0, 2, 3]]

    # Discard invalid triangles
    f = f[(f != -1).all(axis=-1)]

    v = torch.from_numpy(np.ascontiguousarray(v)).to(device, dtype=torch.float32)
    f = torch.from_numpy(np.ascontiguousarray(f)).to(device, dtype=torch.int64)

    return v, f.to(dtype=torch.int32)