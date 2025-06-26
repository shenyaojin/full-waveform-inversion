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

    Args:
        vel_model (np.ndarray): The current velocity model.
        grid (Grid): The grid geometry.
        source (Source): The source used for the forward model.
        receivers (Receivers): The receiver geometry.
        d_obs (np.ndarray): The observed data for this source.
        nt (int): Number of time steps.
        dt (float): Time step size.
        boundary_width (int): Width of the absorbing boundary.

    Returns:
        np.ndarray: The FWI gradient, same shape as the velocity model.
    """
    print("--- Starting Gradient Calculation ---")

    # --- 1. Run forward simulation to get synthetic data and full wavefield ---
    print("Step 1: Running forward simulation to get u(t)...")
    d_syn, u_all = solve_acoustic(
        vel_model, grid, source, receivers, nt, dt,
        boundary_width=boundary_width, store_wavefield=True
    )

    # --- 2. Compute the data residual ---
    # This is the "adjoint source"
    residual = d_syn - d_obs

    # --- 3. Run adjoint simulation (backward in time) ---
    print("\nStep 2: Running adjoint simulation to get v(t)...")

    # Initialize adjoint wavefields
    v_prev = np.zeros(grid.shape, dtype=np.float32)
    v_curr = np.zeros(grid.shape, dtype=np.float32)
    v_next = np.zeros(grid.shape, dtype=np.float32)

    # Initialize the gradient
    gradient = np.zeros_like(vel_model, dtype=np.float32)

    # Same damping from forward solver
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

    # Convert receiver coords to grid indices
    rec_iz = (receivers.coordinates[:, 0] / grid.dz).astype(int)
    rec_ix = (receivers.coordinates[:, 1] / grid.dx).astype(int)
    vel_squared = vel_model ** 2

    # Time-stepping loop (runs backward from nt-1 to 0)
    for it in reversed(range(nt)):
        laplacian = (
                (v_curr[1:-1, 2:] - 2 * v_curr[1:-1, 1:-1] + v_curr[1:-1, :-2]) / grid.dx ** 2 +
                (v_curr[2:, 1:-1] - 2 * v_curr[1:-1, 1:-1] + v_curr[:-2, 1:-1]) / grid.dz ** 2
        )

        v_prev[1:-1, 1:-1] = (2 * v_curr[1:-1, 1:-1] - v_next[1:-1, 1:-1] +
                              (dt ** 2) * vel_squared[1:-1, 1:-1] * laplacian)

        # Inject the time-reversed residual at receiver locations
        v_prev[rec_iz, rec_ix] += residual[it, :] * (dt ** 2)

        v_prev *= boundary_damp

        # --- 4. Correlate wavefields to update gradient ---
        # The second time derivative of u is approximated by a 3-point stencil
        u_tt = (u_all[it] - 2 * u_all[it - 1] + u_all[it - 2]) / dt ** 2 if it > 1 else np.zeros_like(u_all[0])

        # Zero-lag cross-correlation: update gradient at each time step
        gradient += u_tt * v_curr

        v_next, v_curr = v_curr, v_prev

        if (it + 1) % 500 == 0:
            print(f"  Adjoint modeling: Time step {it + 1}/{nt}")

    print("\nStep 3: Gradient calculation finished.")
    # The FWI gradient is scaled by -2/v^3, but we can often absorb this into the step length.
    # For simplicity, we will omit it here. The direction is the most important part.
    return gradient

