# main.py
import pandas as pd
from datasets import load_dataset
from noise_config import ENABLE_NOISE, NOISE_TYPE, NOISE_LEVEL, NOISE_SEED
from BinarySim import BinarySim  # adjust path if needed

def main():
    # Load one example from GravityBench dataset
    data = load_dataset("GravityBench/GravityBench")["test"][0]

    sim = BinarySim(
        data['simulation_csv_content'], 
        data['task_prompt'], 
        data['expected_units'], 
        data['true_answer'],
        enable_noise=ENABLE_NOISE,
        noise_type=NOISE_TYPE,
        noise_level=NOISE_LEVEL,
        noise_seed=NOISE_SEED
    )

    # Save the raw dataset (with noise if enabled) to CSV
    sim.df.to_csv("simulation_with_noise.csv", index=False)
    print("✅ Saved simulation_with_noise.csv with noise settings applied")

if __name__ == "__main__":
    main()
