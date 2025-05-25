# Gravity-Bench: A Benchmark for AI Discovery of Gravitational Physics

[![Paper](https://img.shields.io/badge/arXiv-2501.18411-B31B1B)](https://arxiv.org/abs/2501.18411)
[![Website](https://img.shields.io/badge/Website-gravitybench.github.io-blue)](https://gravitybench.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Or your chosen license -->

This repository contains the code and benchmark for the ICML 2025 paper: **"Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents"** (arXiv:2501.18411). Visit our project website at [gravitybench.github.io](https://gravitybench.github.io/).

Gravity-Bench is a benchmark that evaluates AI agents on discovering gravitational physics through iterative observation and data analysis. The environment is built around two-body gravitational simulations provided by the [Rebound](https://github.com/hannorein/rebound) library, extended with additional scenarios (e.g., modified gravity, linear drag) to enable testing on out-of-distribution physics. Agents must actively query the environment (observing the positions of stars over time) and reason about the data to solve a range of tasks, such as determining masses, orbital elements, and energy-related properties.

The benchmark is inspired by the historical development of science (the two-body problem of gravitational dynamics) and challenges AI agents with tasks that mirror real-world scientific inquiry, requiring iterative reasoning, dynamic planning, and robust generalization.

## Key Features

-   **Physics-based Environment:**
    Uses high-precision N-body simulations from Rebound, capturing realistic two-body orbital dynamics. Simulations are configured in `scripts/scenarios_config.py` and run via `generalscenarios/Binary.py`. Observational data is stored in `scenarios/sims/` (for agent use) and detailed simulation data for verification is in `scenarios/detailed_sims/`.

-   **Partial Observability Modes:**
    Agents can operate in two modes:
    1.  **Full-Observation (full-obs):** Access the complete simulation data at once (e.g., via `scripts/run_agent.py`).
    2.  **Budgeted Observation (budget-obs):** Collect observations incrementally under a defined observation budget (e.g., via `scripts/run_agent.py --row-wise`). This mode emphasizes strategic observation planning.

-   **Wide Range of Tasks:**
    Over 50 tasks that mirror real astrophysical problems, defined in individual Python files within the `scenarios/` directory (e.g., `mass_star1.py`, `eccentricity.py`). These tasks include:
    -   Inferring stellar masses.
    -   Calculating orbital elements (period, eccentricity, periastron).
    -   Determining system energy.
    -   Handling out-of-distribution scenarios like modified gravity or drag forces, testing generalization capabilities.
    Each task is assigned to multiple simulation variations, configured in `scripts/scenarios_config.json`. See Appendix B of our paper for a detailed description of benchmark problems.

-   **Expert Reference Solutions & Ground Truth:**
    -   **Expert Reference Solutions:** Implemented in each task's `.py` file (e.g., `scenarios/mass_star1.py::Scenario.true_answer(return_empirical=True)`), these solutions emulate an expert's approach using only the observable data (full-obs or budget-obs with uniform sampling). They serve as a strong baseline for agent performance and are generated using `scripts/run_expert_solution.py`.
    -   **Ground-Truth Verification:** The same `true_answer` methods can also return exact values derived from simulation inputs or Rebound's internal calculations (when `verification=True` and `return_empirical=False`). This is used for verifying the correctness of the environment and the empirical solutions.

-   **Agentic Framework:**
    The benchmark is designed for iterative, tool-augmented agents. The provided baseline agent (`agents/tabular_agent.py`) uses:
    -   An `Observe` tool for taking measurements at specific times (`agents/tools/observe_tool.py`).
    -   A Python REPL tool for data analysis (`agents/tools/python_repl_tool.py`).
    -   A `submit_answer` tool for providing the final numeric or boolean answer (`agents/tools/submit_answer_tool.py`).
    This agent is executed using `scripts/run_agent.py` or `scripts/run_agent_range_of_budgets.py` (for varying observation budgets). The agent prompts are detailed in Appendix D of our paper.

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/NolanKoblischke/GravityBench.git # Official Repo URL
    cd GravityBench
    ```

2.  **Install Dependencies:**

    We provide three installation methods. **uv is recommended** for the fastest and most reliable setup:

    ### Option A: uv (Recommended) ⭐
    
    [uv](https://docs.astral.sh/uv/) is a fast Python package manager that automatically handles virtual environments.
    
    ```bash
    # Install uv if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Or: pip install uv
    
    # Install dependencies (automatically creates virtual environment)
    uv sync
    
    # Activate the environment (optional - uv run will auto-activate)
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
    
    Then run commands with `uv run`:
    ```bash
    uv run python scripts/run_agent.py --scenarios period --model gpt-4o
    ```

    ### Option B: Conda/Mamba
    
    If you prefer conda environments:
    ```bash
    # Install dependencies using conda
    conda env create -f environment.yml
    conda activate gravitybench
    ```
    
    Then run commands normally:
    ```bash
    python scripts/run_agent.py --scenarios period --model gpt-4o
    ```

    ### Option C: pip
    
    For traditional pip installation:
    ```bash
    # Create and activate virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```
    
    Then run commands normally:
    ```bash
    python scripts/run_agent.py --scenarios period --model gpt-4o
    ```

3.  **Configure API Keys:**
    
    Create a `.env` file with your API keys:
    ```bash
    # Create .env file
    cat > .env << EOF
    OPENAI_API_KEY=your_openai_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    EOF
    ```
    
    Or copy and edit the example:
    ```bash
    cp .env.example .env
    # Edit .env with your preferred editor
    ```

4.  **Configure Agent Parameters (Optional):**
    Global agent parameters like temperature, max attempts, and API call limits can be adjusted in `config.json`.

5.  **Run a Scenario (Full Observations):**
    This example runs the agent on the `period.py` scenario task using the `gpt-4o` model with full observation access.
    
    ```bash
    # With uv (recommended)
    uv run python scripts/run_agent.py --scenarios period --model gpt-4o
    
    # With conda/pip
    python scripts/run_agent.py --scenarios period --model gpt-4o
    ```
    
    Outputs (a JSON file and an HTML report) will be saved in a new subdirectory under `outputs/`, named like `outputs/gpt-4o_<timestamp>/`.

6.  **Run a Scenario (Budgeted Observations):**
    This example runs the agent on the `max_velocity_star1.py` scenario task, using a Claude model, with a budget of 100 total observations, requesting up to 10 at a time.
    
    ```bash
    # With uv (recommended)
    uv run python scripts/run_agent.py \
        --scenarios max_velocity_star1 \
        --model claude-3-5-sonnet-20241022 \
        --row-wise \
        --max-observations-total 100 \
        --max-observations-per-request 10
    
    # With conda/pip
    python scripts/run_agent.py \
        --scenarios max_velocity_star1 \
        --model claude-3-5-sonnet-20241022 \
        --row-wise \
        --max-observations-total 100 \
        --max-observations-per-request 10
    ```

7.  **Explore Results:**
    After each run, navigate to the `outputs/` (or `outputs_range_of_N/`) directory. Inside the run-specific subfolder (e.g., `outputs/gpt-4o_<timestamp>/`), you will find:
    -   A `<model>_<timestamp>.json` file containing detailed run data.
    -   A `<model>_<timestamp>.html` file, which is an interactive report. Open this in your browser to see a summary and step-by-step logs for each scenario attempt.

    *Note: Running agents, especially with powerful models like GPT-4o or Claude Opus, will incur API costs.*

## Repository Structure

-   **`agents/`**: Contains the main agent logic (`tabular_agent.py`) and its tools (`observe_tool.py`, `python_repl_tool.py`, `submit_answer_tool.py`).
-   **`analysis/`**: Scripts for plotting, generating LaTeX tables, and other analyses to reproduce paper results.
-   **`generalscenarios/`**: Core `Binary.py` class for setting up and running Rebound-based binary star simulations.
-   **`scenarios/`**: Individual Python files defining specific physics tasks (e.g., `mass_star1.py`, `eccentricity.py`). Each file includes the task prompt and the expert reference solution logic.
-   **`scripts/`**:
    -   `scenarios_config.py`: Defines `BinaryScenario` variations (initial conditions, masses, etc.) and provides helper functions (`get_scenario`, `get_all_scenarios`) to instantiate task-specific scenario objects. It reads configurations from `scripts/scenarios_config.json`.
    -   `run_agent.py`: Main script for running an agent on specified tasks. Key options include `--row-wise`, `--max-observations-total`, `--simulate-all`, and `--parallel`.
    -   `run_agent_range_of_budgets.py`: Similar to `run_agent.py` but specialized for scanning multiple observation budgets.
    -   `run_expert_solution.py`: Generates the expert reference solutions for each scenario.
    -   `task_utils.py`: Helper functions used by the expert solutions in `scenarios/` to calculate physical quantities.
    -   `format_utils.py`: Utility functions for formatting agent traces and converting them to JSON/HTML.
-   **`outputs/`** & **`outputs_range_of_N/`**: Default directories for storing raw run results (JSON and HTML).
    -   `outputs/combine_results.py`: Merges multiple JSON outputs from `outputs/` into a single `combined_results.csv` and `chat_histories.csv`.
    -   `outputs_range_of_N/aggregate.py`: Aggregates results from `outputs_range_of_N/` subfolders.
-   **`tests/`**: Unit tests for scenario creation (`test_scenarios.py`) and simulation execution (`test_simulations.py`).
-   **`config.json`**: Global configuration for agent runs (e.g., `TEMPERATURE`, `MAX_ATTEMPTS`, `MAX_TIME_PER_TASK`).
-   **`.env.example`**: Template for the `.env` file, which stores API keys.
-   **`environment.yml`**: Conda environment specification file for dependencies.
-   **`requirements.txt`**: Pip requirements file for dependencies.
-   **`LICENSE`**: Project license file.

## Running Tests

To ensure the environment and core logic are functioning correctly, you can run the unit tests:

```bash
# With uv (recommended)
uv run python -m unittest discover -s tests

# With conda/pip
python -m unittest discover -s tests
```

## Reproducing Paper Results

To reproduce the main results presented in the paper (arXiv:2501.18411), follow these general steps:

1.  **Run Experiments with Different Agents and Settings:**
    Refer to Section 4 of the paper for details on models and configurations tested.
    *   **Full-Observation Mode:**
        ```bash
        # With uv (recommended)
        uv run python scripts/run_agent.py --simulate-all --model gpt-4o-mini-2024-07-18
        uv run python scripts/run_agent.py --simulate-all --model claude-3-5-sonnet-20241022
        
        # With conda/pip
        python scripts/run_agent.py --simulate-all --model gpt-4o-mini-2024-07-18
        python scripts/run_agent.py --simulate-all --model claude-3-5-sonnet-20241022
        ```
    *   **Budgeted-Observation Mode (e.g., 100 observations):**
        ```bash
        # With uv (recommended)
        uv run python scripts/run_agent.py --simulate-all --model gpt-4o-mini-2024-07-18 --row-wise --max-observations-total 100
        
        # With conda/pip
        python scripts/run_agent.py --simulate-all --model gpt-4o-mini-2024-07-18 --row-wise --max-observations-total 100
        ```
    *   **Varying Observation Budgets (for specific scenarios/models, see Figure 2 in paper):**
        ```bash
        # With uv (recommended)
        uv run python scripts/run_agent_range_of_budgets.py --model gpt-4o-2024-11-20 --scenarios max_velocity_star1 periastron --variation "9.6 M, 3.1 M" --variation "3.1 M, 0.18 M, Elliptical, Single Orbit"
        uv run python scripts/run_agent_range_of_budgets.py --model claude-3-5-sonnet-20241022 --scenarios max_velocity_star1 periastron --variation "9.6 M, 3.1 M" --variation "3.1 M, 0.18 M, Elliptical, Single Orbit"
        
        # With conda/pip
        python scripts/run_agent_range_of_budgets.py --model gpt-4o-2024-11-20 --scenarios max_velocity_star1 periastron --variation "9.6 M, 3.1 M" --variation "3.1 M, 0.18 M, Elliptical, Single Orbit"
        python scripts/run_agent_range_of_budgets.py --model claude-3-5-sonnet-20241022 --scenarios max_velocity_star1 periastron --variation "9.6 M, 3.1 M" --variation "3.1 M, 0.18 M, Elliptical, Single Orbit"
        ```

2.  **Generate Expert Baseline Data:**
    This script calculates the performance of the expert-defined empirical solutions.
    ```bash
    # With uv (recommended)
    uv run python scripts/run_expert_solution.py
    
    # With conda/pip
    python scripts/run_expert_solution.py
    ```
    This will create `outputs/expert_baseline_results.csv`.

3.  **Aggregate Raw Experiment Results:**
    After running the agent experiments, combine their JSON outputs:
    ```bash
    # With uv (recommended)
    uv run python outputs/combine_results.py
    
    # With conda/pip
    python outputs/combine_results.py
    ```
    This creates `outputs/combined_results.csv` and `outputs/chat_histories.csv`.

    For results from `run_agent_range_of_budgets.py`:
    ```bash
    # With uv (recommended)
    uv run python outputs_range_of_N/aggregate.py
    
    # With conda/pip
    python outputs_range_of_N/aggregate.py
    ```
    This creates `outputs_range_of_N/aggregated_results.csv`.

4.  **Generate Figures and Tables:**
    Run the analysis scripts located in the `analysis/` directory. These scripts use the aggregated CSV files.
    ```bash
    # With uv (recommended)
    uv run python analysis/table1_scores.py       # For Table 1
    uv run python analysis/table2_massassumption.py # For Table 2 in Appendix
    uv run python analysis/fig_casestudy.py       # For Figure 3 (Traces) - Note: this script uses specific hardcoded observation data.
    uv run python analysis/plot_100_obs_human_performance.py # For Figure 4 (Thresholds)
    uv run python outputs_range_of_N/fig_maxvelocity_and_periastron.py # For Figure 2 (Varying Budgets)
    uv run python outputs_range_of_N/fig_periastron.py # For Figure 5 (Periastron Case Study)
    
    # With conda/pip
    python analysis/table1_scores.py       # For Table 1
    python analysis/table2_massassumption.py # For Table 2 in Appendix
    python analysis/fig_casestudy.py       # For Figure 3 (Traces) - Note: this script uses specific hardcoded observation data.
    python analysis/plot_100_obs_human_performance.py # For Figure 4 (Thresholds)
    python outputs_range_of_N/fig_maxvelocity_and_periastron.py # For Figure 2 (Varying Budgets)
    python outputs_range_of_N/fig_periastron.py # For Figure 5 (Periastron Case Study)
    ```
    The generated figures will typically be saved in `analysis/plots/` or `outputs_range_of_N/plots/`, and LaTeX tables in `analysis/tables/`.

## Citation

If you use Gravity-Bench in your research, please cite our paper:

```bibtex
@misc{koblischke2025gravitybenchv1,
      title={Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents}, 
      author={Nolan Koblischke and Hyunseok Jang and Kristen Menou and Mohamad Ali-Dib},
      year={2025},
      eprint={2501.08411},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.