
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

from noise_config import ENABLE_NOISE, NOISE_TYPE, NOISE_LEVEL, NOISE_SEED
from quick_start import BinarySim


def plot_all_graphs(df, noise_type, noise_level, noise_seed=None, filename=None):
    """Plot all graphs (3D + XY, XZ, YZ projections) in one figure."""

    # If no filename provided, build one based on noise configuration
    if filename is None:
        if noise_type is None or not ENABLE_NOISE:
            filename = "all_star_plots_clean.png"
        else:
            if noise_seed is not None:
                filename = f"all_star_plots_{noise_type}_L{noise_level}_S{noise_seed}.png"
            else:
                filename = f"all_star_plots_{noise_type}_L{noise_level}.png"

    fig = plt.figure(figsize=(15, 12))

    # 3D trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(df['star1_x'], df['star1_y'], df['star1_z'], color='blue', s=10, label='Star 1')
    ax1.scatter(df['star2_x'], df['star2_y'], df['star2_z'], color='red', s=10, label='Star 2')
    ax1.set_title('3D Trajectory')
    ax1.set_axis_off()

    # XY projection
    ax2 = fig.add_subplot(222)
    ax2.scatter(df['star1_x'], df['star1_y'], color='blue', s=10, label='Star 1')
    ax2.scatter(df['star2_x'], df['star2_y'], color='red', s=10, label='Star 2')
    ax2.set_title('XY Plane')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)

    # XZ projection
    ax3 = fig.add_subplot(223)
    ax3.scatter(df['star1_x'], df['star1_z'], color='blue', s=10, label='Star 1')
    ax3.scatter(df['star2_x'], df['star2_z'], color='red', s=10, label='Star 2')
    ax3.set_title('XZ Plane')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.legend()
    ax3.grid(True)

    # YZ projection
    ax4 = fig.add_subplot(224)
    ax4.scatter(df['star1_y'], df['star1_z'], color='blue', s=10, label='Star 1')
    ax4.scatter(df['star2_y'], df['star2_z'], color='red', s=10, label='Star 2')
    ax4.set_title('YZ Plane')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved: {filename}")


def main():
    from datasets import load_dataset

    data = load_dataset("GravityBench/GravityBench")["test"][0]

    # Run simulation
    sim = BinarySim(
        csv=data['simulation_csv_content'],
        task=data['task_prompt'],
        units=data['expected_units'],
        truth=data['true_answer'],
        enable_noise=ENABLE_NOISE,
        noise_type=NOISE_TYPE,
        noise_level=NOISE_LEVEL,
        noise_seed=NOISE_SEED
    )
    sim.run(output_file="binary_sim_results.csv")

    # Report noise configuration
    print("Noise enabled:" if ENABLE_NOISE else "Noise disabled")
    if ENABLE_NOISE:
        print(f"Type: {NOISE_TYPE}, Level: {NOISE_LEVEL}, Seed: {NOISE_SEED}")

    # Plot all graphs in one file, with noise info in filename
    plot_all_graphs(
        sim.df,
        noise_type=NOISE_TYPE,
        noise_level=NOISE_LEVEL,
        noise_seed=NOISE_SEED
    )


if __name__ == "__main__":
    main()
