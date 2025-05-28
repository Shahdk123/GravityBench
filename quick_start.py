#!/usr/bin/env python3
"""
Simple quick start script to run Agent on the first scenario from test.jsonl
Uses budget observations (row-wise) mode with observe, python_repl, and submit_answer tools
"""

import json
import time
import pandas as pd
import numpy as np
import openai
from io import StringIO
import os
from dotenv import load_dotenv
import datasets

# Import the tools
from agents.tools.observe_tool import observe_tool
from agents.tools.python_repl_tool import python_repl_tool, execute_python_repl
from agents.tools.submit_answer_tool import submit_answer_tool, execute_submit_answer

load_dotenv()

class SimpleBinarySim:
    """Minimal binary simulation class that provides CSV data and observation functionality"""
    
    def __init__(self, csv_content, task_prompt, expected_units, true_answer):
        self.df = pd.read_csv(StringIO(csv_content))
        self.task_prompt = task_prompt
        self.expected_units = expected_units
        self.true_answer = float(true_answer)
        self.number_of_observations_requested = 0
        
        # Get time range from the dataset
        min_time = self.df['time'].min()
        max_time = self.df['time'].max()
        total_rows = len(self.df)
        
        # Create the prompt for budget observations
        self.row_wise_prompt = f"""You are a physics discovery agent. Your task is: {task_prompt}

You have access to a binary star system simulation dataset. The data contains time-series observations of two stars with columns:
- time: observation time (ranges from {min_time:.2f} to {max_time:.2f}, {total_rows} total observations)
- star1_x, star1_y, star1_z: position coordinates of star 1 (in meters)
- star2_x, star2_y, star2_z: position coordinates of star 2 (in meters)

Expected answer units: {expected_units}

You can:
1. Use the Observe tool to request specific time observations (up to 10 per request, 100 total). The tool returns a confirmation message.
2. Use PythonREPL to analyze data and perform calculations. You have access to the variable 'row_wise_results' which contains your previous observations.
3. Use submit_answer to provide your final numerical answer

Available packages: numpy (as np), pandas (as pd), scipy, sklearn, statsmodels (as sm)

To access your observations in Python, use:
```python
# Access all your previous observations as a pandas DataFrame
print(row_wise_results.observations.head())  # View first few rows
print(row_wise_results.observations.columns)  # See all columns
```

To start, observe some time points within the range {min_time:.2f} to {max_time:.2f} to understand the binary system dynamics."""

    def observe_row(self, times_requested, max_observations_per_request):
        """Observe specific rows from the dataset"""
        if len(times_requested) > max_observations_per_request:
            return f"Error: Requested {len(times_requested)} observations, but maximum per request is {max_observations_per_request}"
        
        self.number_of_observations_requested += len(times_requested)
        
        # Find closest times in the dataset
        observed_rows = []
        for requested_time in times_requested:
            closest_idx = (self.df['time'] - requested_time).abs().idxmin()
            observed_rows.append(self.df.iloc[closest_idx])
        
        result_df = pd.DataFrame(observed_rows)
        return result_df

class SimpleRowWiseResults:
    """Simple container to store observations for the agent"""
    
    def __init__(self):
        self.observations = pd.DataFrame()
    
    def add_observations(self, df):
        """Add new observations from dataframe"""
        if self.observations.empty:
            self.observations = df.copy()
        else:
            self.observations = pd.concat([self.observations, df], ignore_index=True)
            # Remove duplicate rows based on time
            self.observations = self.observations.drop_duplicates(subset=['time']).reset_index(drop=True)

class SimpleEnvironment:
    """Minimal environment wrapper"""
    
    def __init__(self, binary_sim):
        self.binary_sim = binary_sim

class SimpleAgent:
    """Minimal agent that uses the three tools"""
    
    def __init__(self, environment, model, max_observations_total=100, max_observations_per_request=10):
        self.environment = environment
        self.model = model
        self.max_observations_total = max_observations_total
        self.max_observations_per_request = max_observations_per_request
        
        # Create row_wise_results container
        self.row_wise_results = SimpleRowWiseResults()
        
        # Available packages for Python REPL
        self.available_packages = {
            "np": np, 
            "pd": pd,
            "scipy": __import__('scipy'),
            "sklearn": __import__('sklearn'),
            "sm": __import__('statsmodels.api', fromlist=['api']),
            "row_wise_results": self.row_wise_results
        }
        
        # Setup tools
        self.tools = [
            observe_tool(maximum_observations_per_request=self.max_observations_per_request, 
                        metadata={'environment': self.environment}),
            python_repl_tool(_globals=self.available_packages, _locals=self.available_packages, 
                           package_names="numpy pandas scipy sklearn statsmodels"),
            submit_answer_tool()
        ]
        
        # Initialize OpenAI client
        self.client = openai.OpenAI()
        
    def run(self, verbose=True):
        """Run the agent"""
        messages = [
            {"role": "system", "content": self.environment.binary_sim.row_wise_prompt},
            {"role": "user", "content": "Begin your analysis."}
        ]
        
        max_iterations = 20
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            if verbose:
                print(f"\n=== Iteration {iteration} ===")
            
            try:
                # Get response from model
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools
                )
                
                assistant_message = response.choices[0].message
                content = assistant_message.content or ""
                tool_calls = assistant_message.tool_calls
                
                if content and verbose:
                    print(f"Assistant: {content}")
                
                if tool_calls:
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        if verbose:
                            print(f"Tool: {tool_name}")
                            print(f"Args: {tool_args}")
                        
                        # Execute tool
                        if tool_name == "Observe":
                            result_df = self.environment.binary_sim.observe_row(
                                tool_args["times_requested"],
                                self.max_observations_per_request
                            )
                            # Add observations to row_wise_results
                            self.row_wise_results.add_observations(result_df)
                            result = f"Successfully observed {len(result_df)} data points. Data is now available in 'row_wise_results.observations' dataframe with {len(self.row_wise_results.observations)} total observations."
                        elif tool_name == "PythonREPL":
                            result = execute_python_repl(
                                tool_args["input_code"],
                                self.available_packages,
                                self.available_packages
                            )
                        elif tool_name == "submit_answer":
                            result = execute_submit_answer(tool_args["answer"])
                            if verbose:
                                print(f"Final Answer: {result}")
                            return result
                        
                        if verbose:
                            print(f"Result: {result}")
                        
                        messages.append({
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tool_call.id
                        })
                else:
                    # No tool calls, just continue conversation
                    messages.append({"role": "assistant", "content": content})
                    
            except Exception as e:
                print(f"Error: {e}")
                break
        
        print("Max iterations reached without answer submission")
        return None

def main():
    """Main function to run the quick start"""
    print("=== GravityBench Quick Start ===")
    
    # Load first scenario from HuggingFace dataset
    print("Loading first scenario from HuggingFace dataset...")
    dataset = datasets.load_dataset("GravityBench/GravityBench")
    scenario_data = dataset['test'][0] # First scenario from test split
    
    print(f"Scenario: {scenario_data['scenario_name']}")
    print(f"Variation: {scenario_data['variation_name']}")
    print(f"Task: {scenario_data['task_prompt']}")
    print(f"Expected units: {scenario_data['expected_units']}")
    print(f"True answer: {scenario_data['true_answer']}")
    
    # Create simulation and environment
    binary_sim = SimpleBinarySim(
        csv_content=scenario_data['simulation_csv_content'],
        task_prompt=scenario_data['task_prompt'],
        expected_units=scenario_data['expected_units'],
        true_answer=scenario_data['true_answer']
    )
    
    environment = SimpleEnvironment(binary_sim)
    
    # Create and run agent
    print("\nStarting agent...")
    agent = SimpleAgent(environment, model="gpt-4.1")
    
    start_time = time.time()
    result = agent.run(verbose=True)
    end_time = time.time()
    
    # Evaluate result
    print(f"\n=== Results ===")
    print(f"Agent answer: {result}")
    print(f"True answer: {binary_sim.true_answer}")
    print(f"Runtime: {end_time - start_time:.2f} seconds")
    print(f"Observations used: {binary_sim.number_of_observations_requested}/100")
    
    if result is not None:
        percent_error = abs((float(result) - binary_sim.true_answer) / binary_sim.true_answer) * 100
        threshold = scenario_data['budget_obs_threshold_percent']
        correct = percent_error <= threshold
        print(f"Percent error: {percent_error:.2f}%")
        print(f"Threshold: {threshold}%")
        print(f"Correct: {correct}")
    else:
        print("No answer submitted")

if __name__ == "__main__":
    main()
