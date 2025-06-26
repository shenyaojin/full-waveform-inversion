# Shenyao Jin, 2025-06-26
# This script reads a .tar.gz archive containing a SEGY file. I currently test it with marmousi2 dataset.
import numpy as np
import segyio
import tarfile
import os
import tempfile


def convert_targz_segy_to_numpy(targz_path: str) -> np.ndarray | None:
    """
    Finds the first .segy or .sgy file in a .tar.gz archive,
    extracts it to a temporary location, reads its trace data,
    and returns it as a 2D NumPy array. Cleans up the temporary file.

    Args:
        targz_path (str): The file path to the .tar.gz archive.

    Returns:
        np.ndarray | None: A 2D NumPy array containing the seismic trace data,
                            with shape (num_traces, num_samples).
                            Returns None if no SEGY file is found or an error occurs.
    """
    print(f"Opening archive: {targz_path}...")
    try:
        # Open the .tar.gz file for reading with gzip compression
        with tarfile.open(targz_path, 'r:gz') as tar:
            # Iterate through each file in the archive
            for member in tar.getmembers():
                # Check if the file is a regular file and has a SEGY extension
                if member.isfile() and (member.name.lower().endswith('.sgy') or member.name.lower().endswith('.segy')):
                    print(f"Found SEGY file in archive: {member.name}")

                    # Create a temporary directory to extract the file into
                    with tempfile.TemporaryDirectory() as temp_dir:
                        print(f"Extracting '{member.name}' to a temporary directory...")
                        # Use filter='data' to address the DeprecationWarning and enhance security.
                        tar.extract(member, path=temp_dir, filter='data')

                        # Construct the full path to the extracted file
                        extracted_file_path = os.path.join(temp_dir, member.name)
                        print(f"Reading from temporary file: {extracted_file_path}")

                        # Open the extracted SEGY file from the disk
                        with segyio.open(extracted_file_path, ignore_geometry=True) as segyfile:
                            # Use a more robust method to read traces directly.
                            # This reads all traces into a list of arrays, then stacks them.
                            # It is less dependent on complex header geometry than segyio.tools.cube().
                            segy_data = np.stack([trace for trace in segyfile.trace])

                            print(f"Successfully converted {member.name} to NumPy array with shape: {segy_data.shape}")

                            # Return the data. The temporary directory and its contents
                            # will be automatically removed when the 'with' block exits.
                            return segy_data

            # If the loop finishes without finding a SEGY file
            print("No .sgy or .segy file found in the archive.")
            return None

    except tarfile.ReadError:
        print(f"Error: '{targz_path}' is not a valid tar archive or is corrupted.")
        return None
    except FileNotFoundError:
        print(f"Error: The file '{targz_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# --- Main execution block ---
# Note: This block uses the file path you provided.
# It does not create a dummy file anymore.
if __name__ == "__main__":
    # Define the name for our test archive
    # IMPORTANT: Ensure this path is correct relative to where you run the script.
    targz_file = "data/MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz"

    # Run the conversion function on the created archive
    print(f"\n--- Running conversion on '{targz_file}' ---")
    numpy_array = convert_targz_segy_to_numpy(targz_file)

    # Check the result
    if numpy_array is not None:
        print("\nConversion successful!")
        print(f"Type of returned object: {type(numpy_array)}")
        print(f"Shape of the NumPy array: {numpy_array.shape}")
        print("First 5 samples of the first 3 traces:")
        print(numpy_array[:3, :5])
    else:
        print("\nConversion failed or no SEGY file was found.")