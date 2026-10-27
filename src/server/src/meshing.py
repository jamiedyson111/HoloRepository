import numpy as np
import trimesh
from nibabel.affines import apply_affine
from nibabel.spatialimages import SpatialImage
from skimage.measure import marching_cubes


def meshing(image: SpatialImage) -> trimesh.Trimesh | None:
    # image is a 3d grid of numbers, one per voxel
    # 0 being background, this mask filters bg out
    mask = np.asarray(image.dataobj) > 0

    if not mask.any():
        return None

    # padding to close off surfaces that are close to endges
    padded = np.pad(mask, pad_width=1, constant_values=False)

    # adding triangles based on the cube, maybe add preprocessing
    vertices, faces, _, _ = marching_cubes(
        padded.astype(np.float32),
        level=0.5,
        allow_degenerate=False,
    )
    vertices -= 1.0

    # voxel -> RAS millimetres; niivue renders meshes in this same space,
    # so they line up with the source NIfTI without any further transform
    vertices_mm = apply_affine(image.affine, vertices)

    mesh = trimesh.Trimesh(vertices=vertices_mm, faces=faces, process=True)
    mesh.fix_normals(multibody=True)
    _ = trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=20)
    return mesh
