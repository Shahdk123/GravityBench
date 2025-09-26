# noise_config.py
# Central configuration for simulation noise

ENABLE_NOISE = True  # Set to False to disable noise completely
NOISE_TYPE = 'gaussian'  # Options: 'gaussian', 'linear_growth', 'exponential_growth', 'power_law'
NOISE_LEVEL = 0.1  # Adjust this value to control noise magnitude
NOISE_SEED = 42  # Set to None for random noise, or an integer for reproducible noise
