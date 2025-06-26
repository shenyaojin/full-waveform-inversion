# src/geometry.py
# Classes and functions to define the FWI model geometry,
# sources, and receivers.
# SY Jin, shenyaojin@mines.edu

import numpy as np

class Grid:
    """
    A class to represent the computational grid for the FWI problem.

    Attributes:
        shape (tuple): The number of grid points in each dimension (nz, nx).
        origin (tuple): The physical coordinates of the top-left corner (z, x).
        spacing (tuple): The distance between grid points in each dimension (dz, dx).
    """
    def __init__(self, shape, origin=(0., 0.), spacing=(10., 10.)):
        self.shape = shape
        self.origin = origin
        self.spacing = spacing
        print(f"Grid initialized with shape={self.shape}, spacing={self.spacing}m")

    @property
    def nz(self):
        return self.shape[0]

    @property
    def nx(self):
        return self.shape[1]

    @property
    def dz(self):
        return self.spacing[0]

    @property
    def dx(self):
        return self.spacing[1]


def ricker_wavelet(freq, nt, dt, peak_time=1.0):
    """
    Generates a Ricker wavelet.

    Args:
        freq (float): The peak frequency of the wavelet in Hz.
        nt (int): The number of time samples.
        dt (float): The time step in seconds.
        peak_time (float): The time at which the wavelet peaks.

    Returns:
        np.ndarray: The Ricker wavelet time series.
    """
    t = np.arange(0, nt * dt, dt) - peak_time
    # Mathematical formula for the Ricker wavelet
    y = (1.0 - 2.0 * (np.pi**2) * (freq**2) * (t**2)) * \
        np.exp(-(np.pi**2) * (freq**2) * (t**2))
    return y


class Source:
    """
    A class to represent a seismic source.

    Attributes:
        coordinates (tuple): The (z, x) coordinates of the source.
        wavelet (np.ndarray): The time series of the source wavelet.
    """
    def __init__(self, coordinates, freq, nt, dt, peak_time=1.0):
        self.coordinates = coordinates
        self.wavelet = ricker_wavelet(freq, nt, dt, peak_time)
        print(f"Source initialized at {self.coordinates}m with {freq}Hz Ricker wavelet.")

    @property
    def z_pos(self):
        return self.coordinates[0]

    @property
    def x_pos(self):
        return self.coordinates[1]


class Receivers:
    """
    A class to represent an array of receivers.

    Attributes:
        coordinates (np.ndarray): A 2D array where each row is the (z, x)
                                  coordinate of a receiver.
    """
    def __init__(self, coordinates):
        # Ensure coordinates are a NumPy array for easier indexing
        self.coordinates = np.array(coordinates)
        print(f"Receivers initialized at {self.num_receivers} locations.")

    @property
    def num_receivers(self):
        return self.coordinates.shape[0]