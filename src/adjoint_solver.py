# src/adjoint_solver.py
# The adjoint solver for computing the FWI gradient.

import numpy as np
from src.geometry import Grid, Receivers
from src.forward_solver import solve_acoustic


def compute_gradient(
        vel_model: np.ndarray,
        grid: Grid,
        source,  # The source object from the forward model
        receivers: Receivers,
        d_obs: np.ndarray,  # Observed data
        nt: int,
        dt: float,
        boundary_width: int = 50
):
    """
    Computes the FWI gradient for a single source using the adjoint-state method.
    """
    print("--- Starting Gradient Calculation ---")

    # --- 1. Run forward simulation to get synthetic data and full wavefield ---
    print("Step 1: Running forward simulation to get u(t)...")
    d_syn, u_all = solve_acoustic(
        vel_model, grid, source, receivers, nt, dt,
        boundary_width=boundary_width, store_wavefield=True
    )

    # --- 2. Compute the data residual ---
    residual = d_syn - d_obs

    # --- 3. Run adjoint simulation (backward in time) ---
    print("\nStep 2: Running adjoint simulation to get v(t)...")

    v_prev = np.zeros(grid.shape, dtype=np.float32)
    v_curr = np.zeros(grid.shape, dtype=np.float32)
    v_next = np.zeros(grid.shape, dtype=np.float32)

    gradient = np.zeros_like(vel_model, dtype=np.float32)

    nz, nx = grid.nz, grid.nx
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

    rec_iz = (receivers.coordinates[:, 0] / grid.dz).astype(int)
    rec_ix = (receivers.coordinates[:, 1] / grid.dx).astype(int)
    vel_squared = vel_model ** 2

    imaging_factor = -2.0 / (vel_model ** 3 + 1e-10)

    # Time-stepping loop (runs backward from nt-1 to 0)
    for it in reversed(range(nt)):
        # At the start of this loop, v_curr is the field at time `it`
        # and v_next is the field at `it+1`

        # --- 4. Correlate wavefields at the correct time `it` ---
        if it > 0 and it < nt - 1:
            u_tt = (u_all[it + 1] - 2 * u_all[it] + u_all[it - 1]) / dt ** 2
        else:
            u_tt = np.zeros_like(u_all[0])

        # Correlate u_tt(it) with v(it)
        gradient += imaging_factor * u_tt * v_curr

        # --- Now, calculate the *next* adjoint wavefield (which is v_prev, for time it-1) ---
        laplacian = (
                (v_curr[1:-1, 2:] - 2 * v_curr[1:-1, 1:-1] + v_curr[1:-1, :-2]) / grid.dx ** 2 +
                (v_curr[2:, 1:-1] - 2 * v_curr[1:-1, 1:-1] + v_curr[:-2, 1:-1]) / grid.dz ** 2
        )

        v_prev[1:-1, 1:-1] = (2 * v_curr[1:-1, 1:-1] - v_next[1:-1, 1:-1] +
                              (dt ** 2) * vel_squared[1:-1, 1:-1] * laplacian)

        # Inject the time-reversed residual for the *previous* time step `it-1`
        # Since the loop is reversed, residual[it] corresponds to time `it`
        v_prev[rec_iz, rec_ix] += residual[it, :] * (dt ** 2)
        v_prev *= boundary_damp

        # --- IMPORTANT FIX: Pointer swap at the END of the loop ---
        v_next, v_curr = v_curr, v_prev

        if (it + 1) % 500 == 0:
            print(f"  Adjoint modeling: Time step {it + 1}/{nt}")

    print("\nStep 3: Gradient calculation finished.")
    return gradient
