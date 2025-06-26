# src/core.py
# Main class to define and run an FWI problem.

import numpy as np
from scipy.ndimage import gaussian_filter

# --- Import our custom modules ---
import src.io as io
from src.geometry import Grid, Source, Receivers
from src.forward_solver import solve_acoustic
from src.adjoint_solver import compute_gradient
from src.optimization import gradient_descent_step
from src.plotting import plot_velocity_model, plot_shot_record


class FWIProblem:
    """
    Encapsulates the entire FWI workflow for a given dataset and parameters.
    """

    def __init__(self, config):
        """
        Initializes the FWI problem based on a configuration dictionary.
        This sets up the models and geometry but does NOT run the simulation.
        """
        print("--- Initializing FWI Problem ---")
        self.config = config

        # --- Unpack simulation parameters ---
        self.nt = config['nt']
        self.dt = config['dt']

        # --- Setup models and geometry ---
        self.p_vel_true = io.convert_targz_segy_to_numpy(config['true_model_path'])

        self.grid = Grid(shape=self.p_vel_true.shape, spacing=config['grid_spacing'])

        self.p_vel_initial = self._create_initial_model(sigma=config['initial_model_smoothing'])
        self.p_vel_current = self.p_vel_initial.copy()

        self.source = Source(
            coordinates=config['source_coords'],
            freq=config['source_freq'],
            nt=self.nt,
            dt=self.dt
        )
        self.receivers = Receivers(coordinates=config['receiver_coords'])
        # The observed data will be generated later by calling generate_observed_data()
        self.d_obs = None

    def _create_initial_model(self, sigma=10):
        """
        Creates the starting velocity model by applying a Gaussian smooth filter.
        """
        print(f"Creating initial model by smoothing with sigma={sigma}...")
        initial_model = gaussian_filter(self.p_vel_true, sigma=sigma)
        return initial_model

    def generate_observed_data(self):
        """
        Generates the 'observed' data by running the forward solver with the TRUE model.
        This should be called after the final geometry (cropping, receiver placement) is set.
        """
        print("\n--- Generating Observed Data (using true model) ---")
        self.d_obs = solve_acoustic(
            self.p_vel_true, self.grid, self.source, self.receivers, self.nt, self.dt
        )
        print("Observed data generated.")
        plot_shot_record(self.d_obs, dt=self.dt, title="Observed Data (from True Model)")

    def run_inversion(self, num_iterations):
        """
        The main FWI inversion loop.
        """
        print("\n--- Starting FWI Inversion ---")
        if self.d_obs is None:
            raise RuntimeError("Observed data has not been generated. Call generate_observed_data() first.")

        learning_rate = self.config['learning_rate']

        for i in range(num_iterations):
            print(f"\n--- Inversion Iteration {i + 1}/{num_iterations} ---")

            gradient = compute_gradient(
                vel_model=self.p_vel_current,
                grid=self.grid,
                source=self.source,
                receivers=self.receivers,
                d_obs=self.d_obs,
                nt=self.nt,
                dt=self.dt
            )

            self.p_vel_current = gradient_descent_step(
                model=self.p_vel_current,
                gradient=gradient,
                learning_rate=learning_rate
            )

            plot_velocity_model(self.p_vel_current, grid=self.grid, title=f"Velocity Model - Iteration {i + 1}")

        print("\n--- FWI Inversion Finished! ---")
        return self.p_vel_current
