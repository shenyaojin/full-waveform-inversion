# The core function of FWI.
# Shenyao Jin, shenyaojin@mines.edu

#%% Load libraries
import numpy as np
import src.io as io

#%% Load data
filepath = "data/MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz"
# True velocity model
p_vel_true = io.convert_targz_segy_to_numpy(filepath)
print("True velocity model loaded. Shape:", p_vel_true.shape)