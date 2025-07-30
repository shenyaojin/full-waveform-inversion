# test/104_fwi_test_synthetic.py
# A clean, simple test case for FWI using a synthetic model
# to verify the algorithm's correctness.

import numpy as np

# --- Import our FWI modules ---
from src.core import FWIProblem
from src.plotting import plot_velocity_model


def create_synthetic_models():
    """
    Creates a simple true model (a block) and a smooth initial model.
    """
    print("--- Creating synthetic models for test ---")
    shape = (201, 301)  # nz, nx

    # Create a background model
    true_model = np.full(shape, 2000, dtype=np.float32)

    # Add a high-velocity square in the middle
    z_start, z_end = 90, 110
    x_start, x_end = 140, 160
    true_model[z_start:z_end, x_start:x_end] = 2500

    # The initial model will be created by FWIProblem, so we only need the true one.
    return true_model


def main():
    """
    Main function to configure and run the FWI problem on the synthetic data.
    """
    print("======================================================")
    print("=        FWI Algorithm Correctness Test              =")
    print("======================================================")

    # --- 1. Create the true model ---
    true_model = create_synthetic_models()

    # --- 2. FWI Configuration ---
    config = {
        # Pass the numpy array directly to the FWIProblem class
        'true_model_path': true_model,

        'grid_spacing': (10., 10.),
        'initial_model_smoothing': 15,

        'tmax': 2.0,
        'dt': 0.001,

        'source_freq': 10.0,
        'source_coords': (20., (true_model.shape[1] // 2) * 10.),  # Center source

        'receiver_coords': [],  # Will be set after initialization

        'learning_rate': 5e5,  # Learning rates can be very different for different models
        'num_iterations': 10
    }
    config['nt'] = int(config['tmax'] / config['dt'])

    # --- 3. Initialize FWI Problem ---
    # The class will now correctly handle the numpy array we pass it.
    fwi_problem = FWIProblem(config)

    # --- 4. Define final geometry ---
    grid_width_m = (fwi_problem.grid.nx - 1) * fwi_problem.grid.dx
    rec_z = 20.
    rec_x = np.linspace(0, grid_width_m, num=fwi_problem.grid.nx // 2)
    fwi_problem.receivers.coordinates = np.array([(rec_z, x) for x in rec_x])

    plot_velocity_model(fwi_problem.p_vel_true, grid=fwi_problem.grid, title="True Model (Synthetic)")
    plot_velocity_model(fwi_problem.p_vel_initial, grid=fwi_problem.grid, title="Initial Model (Smoothed)")

    # --- 5. Generate Observed Data and Run Inversion ---
    fwi_problem.generate_observed_data()
    final_model = fwi_problem.run_inversion(config['num_iterations'])

    # --- 6. Plot Final Results ---
    print("\n--- Plotting Final Inversion Results ---")
    plot_velocity_model(final_model, grid=fwi_problem.grid, title="Final Inverted Model")
    plot_velocity_model(fwi_problem.p_vel_true, grid=fwi_problem.grid, title="True Model (For Comparison)")

if __name__ == "__main__":
    main()
