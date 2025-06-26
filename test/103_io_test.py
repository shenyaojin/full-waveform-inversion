#%% Load libraries
from src import io

#%% Load data
datapath = 'data/Vp.segy.tar.gz'
p_vel_true = io.convert_targz_segy_to_numpy(datapath)

#%% Plot the true velocity model
from src.plotting import plot_velocity_model

plot_velocity_model(p_vel_true, title="True P-Wave Velocity Model")