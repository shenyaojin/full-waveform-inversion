# src/plotting.py
# Utility functions for plotting FWI results.

import numpy as np
import matplotlib.pyplot as plt


def plot_velocity_model(vel_model, grid=None, title="Velocity Model"):
    """
    Plots a 2D velocity model.
    """
    plt.figure(figsize=(10, 5))

    extent = None
    if grid:
        extent = [grid.origin[1], grid.origin[1] + grid.nx * grid.dx,
                  grid.origin[0] + grid.nz * grid.dz, grid.origin[0]]

    plt.imshow(vel_model, cmap="viridis", aspect='auto', extent=extent)
    plt.colorbar(label="Velocity (m/s)")
    plt.xlabel("X position (m)")
    plt.ylabel("Z position (m)")
    plt.title(title)
    plt.show()


def plot_shot_record(data, title="Shot Record", dt=0.001):
    """
    Plots a shot record (seismogram).
    """
    plt.figure(figsize=(8, 8))

    # Use a percentile to clip the color range for better visualization
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
    Plots a snapshot of the wavefield.
    """
    plt.figure(figsize=(10, 5))

    # Use a percentile to clip the color range for better visualization
    vmax = np.percentile(wavefield, 99.5)
    vmin = -vmax

    extent = None
    if grid:
        extent = [grid.origin[1], grid.origin[1] + grid.nx * grid.dx,
                  grid.origin[0] + grid.nz * grid.dz, grid.origin[0]]

    plt.imshow(wavefield, cmap="RdBu", vmin=vmin, vmax=vmax, aspect='auto', extent=extent)
    plt.colorbar(label="Amplitude")
    plt.xlabel("X position (m)")
    plt.ylabel("Z position (m)")
    plt.title(title)
    plt.show()
