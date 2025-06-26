# test/101_qc_forward_modeling.py
# QC Test for the forward solver.

import numpy as np
import matplotlib.pyplot as plt

# --- Import our FWI modules ---
from src.geometry import Grid, Source, Receivers
from src.forward_solver import solve_acoustic
from src.plotting import plot_velocity_model, plot_shot_record, plot_wavefield

# --- 1. Set up the model and simulation parameters ---
print("--- Setting up QC test ---")
# Model dimensions
nz, nx = 201, 501  # Number of grid points z, x
dz, dx = 10, 10    # Grid spacing in meters

# Time stepping
tmax = 2.5  # Maximum simulation time in seconds
dt = 0.001  # Time step in seconds
nt = int(tmax / dt) # Number of time steps

# Source properties
source_freq = 10.0 # Peak frequency of the Ricker wavelet in Hz

# --- 2. Create the geometry and a simple velocity model ---

# Create a grid object
grid = Grid(shape=(nz, nx), spacing=(dz, dx))

# Create a simple, constant velocity model
vel_homog = np.full(grid.shape, 2000, dtype=np.float32) # 2000 m/s everywhere
plot_velocity_model(vel_homog, grid=grid, title="Homogeneous Velocity Model (QC)")

# Define the source location and wavelet
source_coords = (dz * 10, dx * (nx // 2)) # z, x position in meters
source = Source(coordinates=source_coords, freq=source_freq, nt=nt, dt=dt, peak_time=0.5)

# Define receiver locations
rec_z = dz * 15 # Constant depth for all receivers
rec_x = np.linspace(dx * 5, dx * (nx - 5), num=nx // 2) # Array of receiver x-positions
receiver_coords = np.array([(rec_z, x) for x in rec_x])
receivers = Receivers(coordinates=receiver_coords)

# --- 3. Run the forward solver ---
print("\n--- Running forward solver for QC ---")
shot_record, final_wavefield = solve_acoustic(
    vel_model=vel_homog,
    grid=grid,
    source=source,
    receivers=receivers,
    nt=nt,
    dt=dt,
    boundary_width=50,
    return_wavefield=True  # Ask the solver to return the final wavefield
)

# --- 4. Plot the results for QC ---
print("\n--- Plotting QC results ---")

# Plot the final wavefield. We expect a circular wavefront.
plot_wavefield(final_wavefield, grid=grid, title="Final Wavefield Snapshot (QC)")

# Plot the shot record. We expect a hyperbolic event (direct arrival).
plot_shot_record(shot_record, dt=dt, title="Shot Record (QC)")

print("\n--- QC test finished ---")
print("Check the plots: The wavefield should be circular and the shot record should show a clear hyperbola.")
