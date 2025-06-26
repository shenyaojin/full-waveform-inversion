# src/forward_solver.py
# The acoustic wave equation forward solver.

import numpy as np
from src.geometry import Grid, Source, Receivers


def solve_acoustic(
        vel_model: np.ndarray,
        grid: Grid,
        source: Source,
        receivers: Receivers,
        nt: int,
        dt: float,
        free_surface: bool = False,
        boundary_width: int = 50,
        return_wavefield: bool = False,
        store_wavefield: bool = False
):
    """
    Solves the 2D acoustic wave equation using a finite-difference method.

    Args:
        vel_model (np.ndarray): The velocity model (nz, nx).
        grid (Grid): The grid object defining the geometry.
        source (Source): The source object.
        receivers (Receivers): The receivers object.
        nt (int): Number of time steps.
        dt (float): Time step size.
        free_surface (bool): If True, applies a free surface boundary condition at z=0.
        boundary_width (int): The width of the absorbing boundary layer in grid points.
        return_wavefield (bool): If True, returns the final wavefield snapshot.
        store_wavefield (bool): If True, stores and returns the entire wavefield history.

    Returns:
        np.ndarray: The recorded seismic data (shot record).
        (Optional) np.ndarray or list: Depends on the flags.
                                       - Final wavefield if return_wavefield is True.
                                       - Full wavefield history if store_wavefield is True.
    """
    print("Starting acoustic forward modeling...")

    # --- Setup ---
    nz, nx = grid.nz, grid.nx
    dz, dx = grid.dz, grid.dx

    # Initialize wavefields
    u_prev = np.zeros(grid.shape, dtype=np.float32)
    u_curr = np.zeros(grid.shape, dtype=np.float32)
    u_next = np.zeros(grid.shape, dtype=np.float32)

    # Storage for the full wavefield history if requested
    u_all = None
    if store_wavefield:
        u_all = np.zeros((nt, nz, nx), dtype=np.float32)

    # Prepare receiver data array
    receiver_data = np.zeros((nt, receivers.num_receivers), dtype=np.float32)

    # Convert coordinates to grid indices
    src_iz, src_ix = int(source.z_pos / dz), int(source.x_pos / dx)
    rec_iz, rec_ix = (receivers.coordinates[:, 0] / dz).astype(int), (receivers.coordinates[:, 1] / dx).astype(int)

    vel_squared = vel_model ** 2

    # --- Absorbing Boundary Conditions (ABC) Setup ---
    boundary_damp = np.ones(grid.shape, dtype=np.float32)
    abs_coeff = 0.005  # Damping coefficient

    for i in range(boundary_width):
        val = np.exp(-(abs_coeff * (boundary_width - i)) ** 2)
        boundary_damp[:, i] *= val
        boundary_damp[:, nx - 1 - i] *= val

    start_z = 1 if free_surface else 0
    for i in range(start_z, boundary_width):
        val = np.exp(-(abs_coeff * (boundary_width - i)) ** 2)
        boundary_damp[i, :] *= val
    for i in range(boundary_width):
        val = np.exp(-(abs_coeff * (boundary_width - i)) ** 2)
        boundary_damp[nz - 1 - i, :] *= val

    # --- Time-stepping loop ---
    for it in range(nt):
        laplacian = (
                (u_curr[1:-1, 2:] - 2 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]) / dx ** 2 +
                (u_curr[2:, 1:-1] - 2 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]) / dz ** 2
        )

        u_next[1:-1, 1:-1] = (2 * u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                              (dt ** 2) * vel_squared[1:-1, 1:-1] * laplacian)

        if it < len(source.wavelet):
            u_next[src_iz, src_ix] += source.wavelet[it] * (dt ** 2)

        u_next *= boundary_damp
        if free_surface: u_next[0, :] = 0.

        receiver_data[it, :] = u_next[rec_iz, rec_ix]

        u_prev, u_curr = u_curr, u_next

        if store_wavefield:
            u_all[it] = u_curr

        if (it + 1) % 500 == 0: print(f"  Forward modeling: Time step {it + 1}/{nt}")

    print("Forward modeling finished.")

    # --- Return appropriate values ---
    if store_wavefield:
        return receiver_data, u_all
    if return_wavefield:
        return receiver_data, u_curr
    return receiver_data
