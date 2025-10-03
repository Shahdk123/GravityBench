# quick_start.py
from io import StringIO
import pandas as pd
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()

from agents.tools.submit_answer_tool import submit_answer_tool, execute_submit_answer
from noise_config import ENABLE_NOISE, NOISE_TYPE, NOISE_LEVEL, NOISE_SEED

class BinarySim:
    def __init__(self, csv, task, units, truth,
                 enable_noise=False, noise_type=None,
                 noise_level=0.0, noise_seed=None):
        self.df = pd.read_csv(StringIO(csv))
        self.truth = float(truth)
        self.req = 0
        t0,t1,n = self.df.time.min(), self.df.time.max(), len(self.df)
        pos_cols = ['star1_x','star1_y','star1_z','star2_x','star2_y','star2_z']

        if enable_noise:
            rng = np.random.default_rng(noise_seed)
            for col in pos_cols:
                if noise_type=="gaussian":
                    self.df[col]+=rng.normal(0,noise_level,len(self.df))
                elif noise_type=="linear_growth":
                    self.df[col]+=np.linspace(0,noise_level,len(self.df))
                elif noise_type=="exponential_growth":
                    self.df[col]*=np.exp(noise_level*np.linspace(0,1,len(self.df)))
                elif noise_type=="power_law":
                    self.df[col]+=noise_level*np.power(np.linspace(1,10,len(self.df)),2)

        noise_info=""
        if enable_noise:
            noise_info=f"\nNote: Observations contain {noise_type} noise (level: {noise_level})."

        self.prompt = (f"You are a physics discovery agent. Task: {task}\nDataset spans {t0:.2f}-{t1:.2f} ({n} rows).{noise_info}\n"
                       "Use Observe(times_requested) to sample <=100 rows (<=10 per call).\n"
                       "Analyse with PythonREPL where 'row_wise_results' stores your samples.\n"
                       f"Expected units: {units}. Submit with submit_answer(answer).")

    def run(self, output_file="simulation_output.csv"):
        self.df.to_csv(output_file,index=False)
        print(f"Simulation results saved to {output_file}")

