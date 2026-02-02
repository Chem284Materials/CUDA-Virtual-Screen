import numpy as np
from math import sin, cos, floor


def read_grid(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    idx = 0
    n = int(lines[idx]); idx += 1

    cx, cy, cz = map(float, lines[idx].split())
    idx += 1  # ligand center (unused)

    x_min, y_min, z_min = map(float, lines[idx].split())
    idx += 1

    dx = float(lines[idx])
    idx += 1

    total_points = int(lines[idx])
    idx += 1

    if total_points != n * n * n:
        raise ValueError("Grid size mismatch")

    grid = np.zeros((n, n, n), dtype=np.float64)

    while idx < len(lines):
        ix, iy, iz, val = lines[idx].split()
        idx += 1
        grid[int(ix), int(iy), int(iz)] = float(val)

    return {
        "n_points_dir": n,
        "x_min": x_min,
        "y_min": y_min,
        "z_min": z_min,
        "grid_spacing": dx,
        "grid": grid
    }


def read_ligand(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    natoms = int(lines[0])
    atoms = []

    for i in range(1, natoms + 1):
        atoms.append(list(map(float, lines[i].split())))

    return np.array(atoms, dtype=np.float64)


def euler_to_matrix(alpha, beta, gamma):
    ca, sa = cos(alpha), sin(alpha)
    cb, sb = cos(beta), sin(beta)
    cg, sg = cos(gamma), sin(gamma)

    R = np.zeros((3,3))

    R[0,0] =  cb*cg
    R[0,1] = -cb*sg
    R[0,2] =  sb

    R[1,0] =  sa*sb*cg + ca*sg
    R[1,1] = -sa*sb*sg + ca*cg
    R[1,2] = -sa*cb

    R[2,0] = -ca*sb*cg + sa*sg
    R[2,1] =  ca*sb*sg + sa*cg
    R[2,2] =  ca*cb

    return R


def transform_ligand(ligand, pose_index):
    centroid = ligand.mean(axis=0)

    alpha = pose_index * 0.37
    beta  = pose_index * 0.51
    gamma = pose_index * 0.29

    R = euler_to_matrix(alpha, beta, gamma)

    tx = 5.0 * sin(pose_index * 0.21)
    ty = 5.0 * cos(pose_index * 0.13)
    tz = 5.0 * sin(pose_index * 0.17 + 0.5)

    out = np.empty_like(ligand)

    for i, a in enumerate(ligand):
        v = a - centroid
        r = R @ v
        out[i] = r + centroid + np.array([tx, ty, tz])

    return out


def trilinear_interp(grid_data, x, y, z):
    g = grid_data["grid"]
    n = grid_data["n_points_dir"]
    dx = grid_data["grid_spacing"]

    x_min = grid_data["x_min"]
    y_min = grid_data["y_min"]
    z_min = grid_data["z_min"]

    # Convert coordinates → floating indices
    i_f = (x - x_min) / dx
    j_f = (y - y_min) / dx
    k_f = (z - z_min) / dx

    i0 = int(floor(i_f))
    j0 = int(floor(j_f))
    k0 = int(floor(k_f))

    # Clamp (exactly like C++)
    i0 = max(0, min(i0, n - 2))
    j0 = max(0, min(j0, n - 2))
    k0 = max(0, min(k0, n - 2))

    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    xd = i_f - i0
    yd = j_f - j0
    zd = k_f - k0

    # Corner values
    C000 = g[i0, j0, k0]
    C100 = g[i1, j0, k0]
    C010 = g[i0, j1, k0]
    C110 = g[i1, j1, k0]
    C001 = g[i0, j0, k1]
    C101 = g[i1, j0, k1]
    C011 = g[i0, j1, k1]
    C111 = g[i1, j1, k1]

    # Interpolate along x
    C00 = C000 * (1 - xd) + C100 * xd
    C01 = C001 * (1 - xd) + C101 * xd
    C10 = C010 * (1 - xd) + C110 * xd
    C11 = C011 * (1 - xd) + C111 * xd

    # Along y
    C0 = C00 * (1 - yd) + C10 * yd
    C1 = C01 * (1 - yd) + C11 * yd

    # Along z
    return C0 * (1 - zd) + C1 * zd


def compute_lj_energy(grid_data, ligand_coords):
    energy = 0.0
    for atom in ligand_coords:
        energy += trilinear_interp(grid_data, atom[0], atom[1], atom[2])
    return energy
