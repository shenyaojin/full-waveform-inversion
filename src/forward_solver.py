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
    """
    print("Starting acoustic forward modeling...")

    # --- Setup ---
    nz, nx = grid.nz, grid.nx
    dz, dx = grid.dz, grid.dx

    # --- CFL Stability Check ---
    max_vel = np.max(vel_model)
    cfl_val = max_vel * dt * np.sqrt(1 / dx ** 2 + 1 / dz ** 2)
    if cfl_val >= 1.0:
        raise ValueError(f"CFL condition not met. Value is {cfl_val:.2f}. "
                         "Decrease dt or increase grid spacing.")
    print(f"  CFL condition is met: {cfl_val:.2f} < 1.0")

    # Initialize wavefields for t-dt, t, and t+dt
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
    abs_coeff = 0.005
    for i in range(boundary_width):
        val = np.exp(-(abs_coeff * (boundary_width - i)) ** 2)
        boundary_damp[:, i] *= val
        boundary_damp[:, nx - 1 - i] *= val
    for i in range(boundary_width):
        val = np.exp(-(abs_coeff * (boundary_width - i)) ** 2)
        boundary_damp[i, :] *= val
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

        # --- IMPORTANT FIX: Use a circular pointer swap for wavefields ---
        # This is the correct and efficient way to advance the wavefields in time
        # without overwriting data needed for the next step.
        u_prev, u_curr, u_next = u_curr, u_next, u_prev

        # --- IMPORTANT FIX: Store a copy of the wavefield ---
        # If we just store u_curr, we store a reference that will change.
        if store_wavefield:
            u_all[it] = u_curr.copy()

        if (it + 1) % 500 == 0: print(f"  Forward modeling: Time step {it + 1}/{nt}")

    print("Forward modeling finished.")

    # --- Sanity Check ---
    if np.allclose(receiver_data, 0):
        print(
            "\nWARNING: The recorded data is all zero. Check simulation parameters (tmax, source/receiver locations).")

    # --- Return appropriate values ---
    if store_wavefield:
        return receiver_data, u_all
    if return_wavefield:
        return receiver_data, u_curr
    return receiver_data
