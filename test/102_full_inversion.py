# test/102_full_inversion.py
# A full FWI example using the Marmousi model.

import numpy as np

# --- Import our FWI modules ---
from src.core import FWIProblem
from src.plotting import plot_velocity_model


def main():
    """
    Main function to configure and run the FWI problem.
    """
    print("======================================================")
    print("=         Full-Wave Inversion Demo Script            =")
    print("======================================================")

    # --- 1. FWI Configuration ---
    config = {
        'true_model_path': 'data/MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz',
        'grid_spacing': (10., 10.),
        'initial_model_smoothing': 25,
        'tmax': 4.0,
        'dt': 0.001,
        'source_freq': 5.0,
        'source_coords': (20., 0.),  # Placeholder, will be updated
        'receiver_coords': [],  # Initialize with an empty list
        'learning_rate': 200,
        'num_iterations': 5
    }
    config['nt'] = int(config['tmax'] / config['dt'])

    # --- 2. Initialize FWI Problem (without running simulation yet) ---
    fwi_problem = FWIProblem(config)

    # --- 3. Crop the model and define final geometry ---
    print("\n--- Defining final geometry for the inversion ---")

    # Let's take a sensible window of the Marmousi model
    # Original shape is (13601, 2801)
    z_start, z_end = 0, 301  # A 3km deep model (301 points)
    x_start, x_end = 1000, 2001  # A 10km wide model (1001 points)

    true_model_cropped = fwi_problem.p_vel_true[z_start:z_end, x_start:x_end].copy()
    initial_model_cropped = fwi_problem.p_vel_initial[z_start:z_end, x_start:x_end].copy()

    # Update the problem attributes with the cropped models
    fwi_problem.p_vel_true = true_model_cropped
    fwi_problem.p_vel_initial = initial_model_cropped
    fwi_problem.p_vel_current = initial_model_cropped.copy()

    # Update the grid object to match the new, smaller model size
    fwi_problem.grid.shape = fwi_problem.p_vel_true.shape

    # --- Correctly define the spatial extent for receivers ---
    # The last grid point in the x-direction is at (nx - 1) * dx
    grid_width_m = (fwi_problem.grid.nx - 1) * fwi_problem.grid.dx

    # Place source in the middle of the cropped model
    fwi_problem.source.coordinates = (20., grid_width_m / 2)

    # Place receivers across the surface of the cropped model, staying within bounds
    rec_z = 20.
    rec_x = np.linspace(0, grid_width_m, num=fwi_problem.grid.nx // 2)
    fwi_problem.receivers.coordinates = np.array([(rec_z, x) for x in rec_x])

    print(f"Model cropped to shape: {fwi_problem.grid.shape}")
    # Fixed the typo here: coordinatas -> coordinates
    print(f"Source placed at: {fwi_problem.source.coordinates} m")
    print(f"Receivers placed at {fwi_problem.receivers.num_receivers} locations.")

    # Plot the true and initial models that will actually be used
    plot_velocity_model(fwi_problem.p_vel_true, grid=fwi_problem.grid, title="True Velocity Model (Cropped)")
    plot_velocity_model(fwi_problem.p_vel_initial, grid=fwi_problem.grid, title="Initial Velocity Model (Smoothed)")

    # --- 4. Generate Observed Data (NOW that geometry is set) ---
    fwi_problem.generate_observed_data()

    # --- 5. Run the Inversion ---
    final_model = fwi_problem.run_inversion(config['num_iterations'])

    # --- 6. Plot Final Results ---
    print("\n--- Plotting Final Inversion Results ---")
    plot_velocity_model(final_model, grid=fwi_problem.grid, title="Final Inverted Velocity Model")
    plot_velocity_model(fwi_problem.p_vel_true, grid=fwi_problem.grid, title="True Velocity Model (For Comparison)")

    print("\n======================================================")
    print("=                 FWI Demo Finished                  =")
    print("======================================================")


if __name__ == "__main__":
    main()
