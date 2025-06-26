# src/plotting.py
# Utility functions for plotting FWI results.

import numpy as np
import matplotlib.pyplot as plt


def plot_velocity_model(vel_model, grid=None, title="Velocity Model"):
    """
    Plots a 2D velocity model, ensuring correct orientation.
    """
    plt.figure(figsize=(10, 5))

    # Transpose the model so that depth is the vertical axis.
    model_to_plot = vel_model.T

    # Define the plot extent based on grid dimensions if available
    extent = None
    if grid:
        extent = [grid.origin[1], grid.origin[1] + grid.nx * grid.dx,
                  grid.origin[0], grid.origin[0] + grid.nz * grid.dz]

    plt.imshow(model_to_plot[::-1], cmap="viridis", aspect='auto', extent=extent)
    plt.colorbar(label="Velocity (m/s)")
    plt.xlabel("X position (m)")
    plt.ylabel("Z position (m)")
    plt.title(title)

    # --- IMPORTANT FIX ---
    # Invert the Y-axis so that depth (Z) increases downwards.
    plt.gca().invert_yaxis()

    plt.show()


def plot_shot_record(data, title="Shot Record", dt=0.001):
    """
    Plots a shot record (seismogram).
    """
    plt.figure(figsize=(8, 8))

    vmax = np.percentile(data, 99)
    vmin = -vmax

    extent = [0, data.shape[1], data.shape[0] * dt, 0]

    plt.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, aspect='auto', extent=extent)
    plt.xlabel("Receiver Index")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.show()


def plot_wavefield(wavefield, grid=None, title="Wavefield Snapshot"):
    """
    Plots a snapshot of the wavefield, ensuring correct orientation.
    """
    plt.figure(figsize=(10, 5))

    # Wavefield is also (nx, nz), so we transpose it.
    wavefield_to_plot = wavefield.T

    vmax = np.percentile(wavefield_to_plot, 99.5)
    vmin = -vmax

    extent = None
    if grid:
        extent = [grid.origin[1], grid.origin[1] + grid.nx * grid.dx,
                  grid.origin[0] + grid.nz * grid.dz, grid.origin[0]]

    plt.imshow(wavefield_to_plot, cmap="RdBu", vmin=vmin, vmax=vmax, aspect='auto', extent=extent)
    plt.colorbar(label="Amplitude")
    plt.xlabel("X position (m)")
    plt.ylabel("Z position (m)")
    plt.title(title)

    # --- IMPORTANT FIX ---
    # Also invert the Y-axis for the wavefield plot.
    plt.gca().invert_yaxis()

    plt.show()
