# src/io.py
# Shenyao Jin, 2025-06-26
# This script reads a .tar.gz archive containing a SEGY file.

import numpy as np
import segyio
import tarfile
import os
import tempfile
# Import the tools submodule from segyio
from segyio import tools

def convert_targz_segy_to_numpy(targz_path: str) -> np.ndarray | None:
    """
    Finds the first .segy or .sgy file in a .tar.gz archive,
    extracts it, and reads its data into a NumPy array using segyio.tools.cube.

    Args:
        targz_path (str): The file path to the .tar.gz archive.

    Returns:
        np.ndarray | None: A NumPy array containing the seismic data.
    """
    print(f"Opening archive: {targz_path}...")
    try:
        with tarfile.open(targz_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile() and (member.name.lower().endswith('.sgy') or member.name.lower().endswith('.segy')):
                    print(f"Found SEGY file in archive: {member.name}")

                    with tempfile.TemporaryDirectory() as temp_dir:
                        print(f"Extracting '{member.name}' to a temporary directory...")
                        tar.extract(member, path=temp_dir, filter='data')

                        extracted_file_path = os.path.join(temp_dir, member.name)
                        print(f"Reading from temporary file: {extracted_file_path}")

                        # --- IMPORTANT FIX ---
                        # Use segyio.tools.cube to read the entire data volume at once.
                        # This is more robust for gridded data like velocity models.
                        # We also convert it to a standard numpy array from a memmap.
                        segy_data = np.array(tools.cube(extracted_file_path))

                        print(f"Successfully converted {member.name} to NumPy array with shape: {segy_data.shape}")
                        return segy_data

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


if __name__ == "__main__":
    # Path to the complex, true velocity model
    targz_file = "data/Vp.segy.tar.gz"

    print(f"\n--- Running conversion on '{targz_file}' ---")
    numpy_array = convert_targz_segy_to_numpy(targz_file)

    if numpy_array is not None:
        print("\nConversion successful!")
        print(f"Shape of the NumPy array: {numpy_array.shape}")
    else:
        print("\nConversion failed.")

