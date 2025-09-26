from agents.tools.submit_answer_tool import submit_answer_tool, execute_submit_answer
from noise_config import ENABLE_NOISE, NOISE_TYPE, NOISE_LEVEL, NOISE_SEED
import pandas as pd
import numpy as np
from io import StringIO
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

class BinarySim:
    def __init__(self, csv, task, units, truth, 
                 enable_noise=False, noise_type=None, 
                 noise_level=0.0, noise_seed=None):
        self.df = pd.read_csv(StringIO(csv))
        self.truth = float(truth)
        self.req = 0  # total observations used
        t0, t1, n = self.df.time.min(), self.df.time.max(), len(self.df)

        # Add noise if enabled
        if enable_noise:
            rng = np.random.default_rng(noise_seed)
            if noise_type == "gaussian":
                self.df["value"] += rng.normal(0, noise_level, size=len(self.df))
            elif noise_type == "linear_growth":
                self.df["value"] += np.linspace(0, noise_level, len(self.df))
            elif noise_type == "exponential_growth":
                self.df["value"] *= np.exp(noise_level * np.linspace(0, 1, len(self.df)))
            elif noise_type == "power_law":
                self.df["value"] += noise_level * np.power(np.linspace(1, 10, len(self.df)), 2)

        noise_info = ""
        if enable_noise:
            noise_info = f"\nNote: Observations contain {noise_type} noise (level: {noise_level})."
        
        self.prompt = (
            f"You are a physics discovery agent. Your task is: {task}\n\n"
            f"Dataset spans {t0:.2f}–{t1:.2f} with {n} rows.{noise_info}\n"
            "Use Observe(times_requested) to sample <=100 rows (<=10 per call).\n"
            "Analyse with PythonREPL where 'row_wise_results' stores your samples.\n"
            f"Expected answer units: {units}. Submit with submit_answer(answer)."
        )

    def run(self, output_file="simulation_output.csv"):
        """Run the simulation and save results to CSV."""
        # Save dataset (possibly noisy) to CSV
        self.df.to_csv(output_file, index=False)
        print(f"Simulation results saved to {output_file}")


def main():
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
    
    # Run and output CSV
    sim.run(output_file="binary_sim_results.csv")

    # Print noise configuration
    if ENABLE_NOISE:
        print(f"Noise: {NOISE_TYPE} with level {NOISE_LEVEL}" + 
              (f" (seed: {NOISE_SEED})" if NOISE_SEED is not None else ""))
    else:
        print("Noise: Disabled")


if __name__ == "__main__":
    main()
