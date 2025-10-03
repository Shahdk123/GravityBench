# Binary.py
import numpy as np
import rebound
import csv
import pandas as pd
from typing import Union, List
from scipy.interpolate import CubicSpline

class RowWiseResults:
    """Stores and manages observational data collected through row-wise access"""
    def __init__(self):
        self.df = pd.DataFrame(columns=['time', 'star1_x', 'star1_y', 'star1_z',
                                        'star2_x', 'star2_y', 'star2_z'])

class Binary:
    """Handles binary star system simulation and observations"""
    def __init__(self, star1_mass, star2_mass, star1_pos, star2_pos, star1_momentum,
                 star2_momentum, maxtime, max_observations, max_observations_per_request,
                 filename, prompt, final_answer_units, drag_tau=None,
                 mod_gravity_exponent=None, units=('m', 's', 'kg'), skip_simulation=False,
                 enable_noise=False, noise_type=None, noise_level=0.0, noise_seed=None):
        
        self.star1_mass = star1_mass
        self.star2_mass = star2_mass
        self.star1_pos = star1_pos
        self.star2_pos = star2_pos
        self.star1_momentum = star1_momentum
        self.star2_momentum = star2_momentum
        self.maxtime = maxtime
        self.max_observations = max_observations
        self.max_observations_per_request = max_observations_per_request
        self.number_of_observations_requested = 0
        self.row_wise_results = RowWiseResults()
        self.filename = filename
        self.final_answer_units = final_answer_units
        self.drag_tau = drag_tau
        self.mod_gravity_exponent = mod_gravity_exponent
        self.units = units
        self.enable_noise = enable_noise
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.noise_seed = noise_seed
        if noise_seed is not None:
            np.random.seed(noise_seed)

        # Initialize REBOUND
        self.sim = rebound.Simulation()
        self.sim.units = self.units
        self.sim.G = 6.67430e-11 if units == ('m','s','kg') else 4*np.pi**2 if units==('yr','AU','Msun') else 6.67430e-8

        if not skip_simulation:
            self.simulate(drag_tau=drag_tau, mod_gravity_exponent=mod_gravity_exponent)

        # Load generated CSV data
        self.df = pd.read_csv(f"scenarios/sims/{self.filename}.csv")
        self.task = prompt

    def _generate_noise(self, time):
        if not self.enable_noise or self.noise_level == 0:
            return 0.0
        if self.noise_type == 'gaussian':
            return np.random.normal(0, self.noise_level)
        elif self.noise_type == 'linear_growth':
            return np.random.normal(0, self.noise_level*time)
        elif self.noise_type == 'exponential_growth':
            return np.random.normal(0, self.noise_level*np.exp(time))
        elif self.noise_type == 'power_law':
            return np.random.normal(0, self.noise_level*time**1.05)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def simulate(self, drag_tau=None, mod_gravity_exponent=None):
        """Run N-body simulation and save positions to CSV"""
        self.sim = rebound.Simulation()
        self.sim.integrator = "whfast"
        self.sim.units = self.units

        # Time step
        r = np.linalg.norm(np.array(self.star2_pos)-np.array(self.star1_pos))
        M = self.star1_mass + self.star2_mass
        T = 2*np.pi*np.sqrt(r**3/(self.sim.G*M))
        self.sim.dt = T/5000
        total_steps = int(self.maxtime/self.sim.dt)
        if total_steps < 1000 or mod_gravity_exponent:
            self.sim.dt = self.maxtime/5000

        # Add stars
        self.sim.add(m=self.star1_mass, x=self.star1_pos[0], y=self.star1_pos[1], z=self.star1_pos[2],
                     vx=self.star1_momentum[0]/self.star1_mass,
                     vy=self.star1_momentum[1]/self.star1_mass,
                     vz=self.star1_momentum[2]/self.star1_mass)
        self.sim.add(m=self.star2_mass, x=self.star2_pos[0], y=self.star2_pos[1], z=self.star2_pos[2],
                     vx=self.star2_momentum[0]/self.star2_mass,
                     vy=self.star2_momentum[1]/self.star2_mass,
                     vz=self.star2_momentum[2]/self.star2_mass)

        # Output CSV
        csv_file_positions = f"scenarios/sims/{self.filename}.csv"
        with open(csv_file_positions, 'w', newline='') as file_positions:
            writer_positions = csv.writer(file_positions)
            header_positions = ['time','star1_x','star1_y','star1_z','star2_x','star2_y','star2_z']
            writer_positions.writerow(header_positions)

            time_passed = 0
            while time_passed < self.maxtime:
                self.sim.integrate(self.sim.t+self.sim.dt)
                time_passed += self.sim.dt

                p1 = self.sim.particles[0]
                p2 = self.sim.particles[1]

                if self.enable_noise:
                    noise = self._generate_noise(time_passed)
                    p1x,p1y,p1z = p1.x+noise, p1.y+noise, p1.z+noise
                    p2x,p2y,p2z = p2.x+noise, p2.y+noise, p2.z+noise
                else:
                    p1x,p1y,p1z = p1.x,p1.y,p1.z
                    p2x,p2y,p2z = p2.x,p2.y,p2.z

                writer_positions.writerow([time_passed,p1x,p1y,p1z,p2x,p2y,p2z])

    def observe_row(self, times_requested: Union[float,List[float]], maximum_observations_per_request: int) -> str:
        if not isinstance(times_requested,list):
            times_requested = [times_requested]
        if len(times_requested)>maximum_observations_per_request:
            return f"Max {maximum_observations_per_request} per request"

        df = pd.read_csv(f"scenarios/sims/{self.filename}.csv")
        remaining = self.max_observations - self.number_of_observations_requested
        to_process = min(len(times_requested), remaining)
        times_to_process = times_requested[:to_process]
        self.number_of_observations_requested += to_process

        observations=[]
        for t in times_to_process:
            if t<0 or t>self.maxtime*1.01:
                observations.append([t]+[None]*6)
                continue
            rows = df.iloc[(df['time']-t).abs().argsort()[:4]].sort_values('time')
            cs = {col:CubicSpline(rows['time'], rows[col].values) for col in df.columns if col!='time'}
            obs = [t]+[cs[col](t) for col in df.columns if col!='time']
            observations.append(obs)

        self.row_wise_results.df = pd.concat([self.row_wise_results.df,pd.DataFrame(observations, columns=self.row_wise_results.df.columns)],ignore_index=True)
        return f"{to_process} observations added. Remaining: {self.max_observations-self.number_of_observations_requested}"

