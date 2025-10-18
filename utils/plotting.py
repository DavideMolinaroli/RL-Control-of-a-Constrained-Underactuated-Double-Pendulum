import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
import csv
# matplotlib.use("TkAgg")

def plot(thetas1, thetas2, omegas1, omegas2, torques, rewards, desired_reward,dt, t_final, model_name):
    # Unwrap angles
    thetas1 = np.unwrap(thetas1)
    thetas2 = np.unwrap(thetas2)

    max_steps = int(t_final / dt) + 1
    time_ticks = [i for i in range(0, max_steps, int(1 / dt))]
    time_labels = [int(i / (1 / dt)) for i in time_ticks]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 5), sharex=True)

    # Plot angles
    ax1.plot(thetas1, color="b", label="ϑ1")
    ax1.plot(thetas2, color="g", label="ϑ2")
    ax1.axhline(y=4*np.pi, color="r", linestyle="--", linewidth=1, label="4π")
    ax1.set_ylabel("Angles [rad]")
    ax1.set_yticks([k*np.pi for k in range(-1, 6)])
    ax1.set_yticklabels([f"{k}π" for k in range(-1, 6)])
    ax1.legend(loc="lower right")
    ax1.grid()
    steps_per_sec = int(1 / dt)

    # Slice last second
    axins = inset_axes(ax1, width="20%", height="40%", loc="lower center")
    axins.plot(thetas1[-steps_per_sec:], color="b", linewidth=0.75)
    axins.plot(thetas2[-steps_per_sec:], color="g", linewidth=0.75)
    axins.axhline(y=2*np.pi, color="k", linestyle="--", linewidth=0.75)
    axins.axhline(y=3*np.pi, color="k", linestyle="--", linewidth=0.75)

    # Force x-axis to show "14" to "15"
    axins.set_xlim(0, steps_per_sec-1)
    axins.set_xticks([0, steps_per_sec//2, steps_per_sec-1])
    axins.set_xticklabels(["14", "14.5", "15"])

    # Custom y ticks for the inset
    axins.set_yticks([1.75*np.pi, 2*np.pi, 3*np.pi, 3.5*np.pi])
    axins.set_yticklabels(["", "2π", "3π", ""])

    # Plot angular velocities
    ax2.plot(omegas1, color="b", label="ω1")
    ax2.plot(omegas2, color="g", label="ω2")
    ax2.set_ylabel("Velocities [rad/s]")
    ax2.legend()
    ax2.legend(loc="lower right")
    ax2.grid()

    # Plot torque
    ax3.plot(torques, color="m", label="Torque")
    ax3.set_ylabel("Torque [Nm]")
    ax3.grid()

    # Plot rewards
    ax4.plot(rewards, color="c", label="Reward")
    # ax4.axhline(y=desired_reward, color="r", linestyle="--", label=f"Desired Reward")
    ax4.set_ylabel("Reward")
    ax4.set_xlabel("Time [s]")
    ax4.grid()

    # Ticks
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels)

    # Save single image
    plt.tight_layout()
    plt.savefig(f"{model_name}_state_evolution.pdf")
    plt.close()

def plot_angles(thetas1, thetas2, dt, t_final, model_name):
    # Unwrap angles
    thetas1 = np.unwrap(thetas1)
    thetas2 = np.unwrap(thetas2)

    max_steps = int(t_final / dt) + 1
    time_ticks = [i for i in range(0, max_steps, int(1 / dt))]
    time_labels = [int(i / (1 / dt)) for i in time_ticks]

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(thetas1, color="b", label="ϑ1")
    ax.plot(thetas2, color="g", label="ϑ2")

    # Horizontal reference line at 4π
    ax.axhline(y=4*np.pi, color="r", linestyle="--", linewidth=1, label="4π")

    # Axis labels and ticks
    ax.set_ylabel("Angles [rad]")
    ax.set_xlabel("Time [s]")
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)
    ax.set_yticks([k*np.pi for k in range(-1, 6)])
    ax.set_yticklabels([f"{k}π" for k in range(-1, 6)])

    ax.legend(loc="lower right")
    ax.grid()

    plt.tight_layout()
    plt.savefig(f"{model_name}_angles.pdf")
    plt.close()

def rewards_plot(rewards, robot, desired_reward, dt, t_final, model_name):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(rewards)
    max_steps = int(t_final / dt) + 1
    ax.set_ylabel("Reward")
    ax.set_xlabel("Time Step")
    # ax.axhline(y=desired_reward, color="r", linestyle="--", label=f"Desired Reward")
    ax.set_xticks([i for i in range(0, max_steps, int(1 / dt))])
    ax.set_xticklabels([i / (1 / dt) for i in range(0, max_steps, int(1 / dt))])
    ax.grid()
    fig.suptitle(f"{robot.capitalize()} Rewards")
    plt.tight_layout()
    plt.savefig(f"{model_name}_rewards.pdf")
    plt.close()

def energy_plot(kinetic_energy, potential_energy, robot, dt, t_final):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(kinetic_energy, label="Kinetic Energy")
    ax.plot(potential_energy, label="Potential Energy")
    max_steps = int(t_final / dt) + 1
    ax.set_ylabel("Energy [J]")
    ax.set_xlabel("Time")
    ax.set_xticks([i for i in range(0, max_steps, int(1 / dt))])
    ax.set_xticklabels([i / (1 / dt) for i in range(0, max_steps, int(1 / dt))])
    ax.legend()
    ax.grid()
    fig.suptitle(f"{robot.capitalize()} Energy")
    plt.savefig("energy.pdf")
    plt.close()

def plot_learning_curves(path_str):
    # Extract folder_name from "folder_name/model_type"
    folder_name = path_str.split("/")[0]

    # Build csv and pdf file paths
    csv_files = {
        "train_mean_reward": f"tensorboard_data/{folder_name}_train_mean_return.csv",
        "eval_reward": f"tensorboard_data/{folder_name}_eval_return.csv"
    }

    for key, csv_path in csv_files.items():
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            continue

        steps = []
        values = []

        # Read csv
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(float(row["Step"])))
                values.append(float(row["Value"]))

        if not steps or not values:
            print(f"CSV file is empty or invalid: {csv_path}")
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(steps, values, color="y")

        ax.set_xlabel("Step")
        ax.set_ylabel(key.replace("_", " ").capitalize())
        ax.grid()

        plt.tight_layout()

        # Save plot
        pdf_path = f"tensorboard_data/{folder_name}_{key}.pdf"
        plt.savefig(pdf_path)
        plt.close()

def plot_learning_curves_together(path_str):
    # Extract folder_name from "folder_name/model_type"
    folder_name = path_str.split("/")[0]

    # Build csv file paths
    csv_files = {
        "train_mean_return": f"tensorboard_data/{folder_name}_train_mean_return.csv",
        "eval_return": f"tensorboard_data/{folder_name}_eval_return.csv"
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 2))

    for key, csv_path in csv_files.items():
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            continue

        steps = []
        values = []

        # Read csv manually
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(float(row["Step"])))
                values.append(float(row["Value"]))

        if not steps or not values:
            print(f"CSV file is empty or invalid: {csv_path}")
            continue

        # Plot each curve
        label = key.replace("_", " ").capitalize()
        color = "y" if "train" in key else "g"
        ax.plot(steps, values, label=label, color=color)

    # Style
    ax.set_xlabel("Step")
    ax.set_ylabel("Return")
    ax.grid()
    ax.legend(loc="lower right")

    plt.tight_layout()

    # Save image
    pdf_path = f"tensorboard_data/{folder_name}_learning_curves.pdf"
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved combined plot: {pdf_path}")

def plot_thetas_stacked(thetas_files, dt=0.01, t_final=5.0, pdf_path="theta_comparison.pdf"):
    """
    thetas_files: list of CSV files containing columns: theta1, theta2
                  Each file corresponds to tau_d1, tau_d2, tau_d3
    dt: time step (s)
    t_final: final time (s)
    pdf_path: output PDF file
    """

    labels1 = ["ϑ1, Loose", "ϑ1, Strict", "ϑ1, Compromise"]
    labels2 = ["ϑ2, Loose", "ϑ2, Strict", "ϑ2, Compromise"]
    colors = ["b","y","g"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3), sharex=True)

    for idx, file_path in enumerate(thetas_files):
        if not os.path.exists(file_path):
            print(f"CSV file not found: {file_path}")
            continue

        thetas1, thetas2 = [], []
        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                thetas1.append(float(row["theta1"]))
                thetas2.append(float(row["theta2"]))

        thetas1 = np.unwrap(np.array(thetas1))
        thetas2 = np.unwrap(np.array(thetas2))

        # Generate time vector
        time = np.arange(0, dt*len(thetas1), dt)

        color = colors[idx] 
        ax1.plot(time, thetas1, color=color, label=labels1[idx])
        ax2.plot(time, thetas2, color=color, label=labels2[idx])

    # Axis labels
    ax1.axhline(y=-4*np.pi, color="r", linestyle="--", linewidth=1, label="4π")
    ax2.axhline(y=-4*np.pi, color="r", linestyle="--", linewidth=1, label="4π")
    ax2.axhline(y=4*np.pi, color="r", linestyle="--", linewidth=1)

    ax2.set_xlabel("Time [s]")
    ax1.set_ylabel("ϑ1 [rad]")
    ax2.set_ylabel("ϑ2 [rad]")
    ax1.set_yticks([k*np.pi for k in range(-5, 3)])
    ax2.set_yticks([k*np.pi for k in range(-5, 7,2)])
    ax1.set_yticklabels([f"{k}π" for k in range(-5, 3)])
    ax2.set_yticklabels([f"{k}π" for k in range(-5, 7,2)])
    ax1.set_ylim([-5*np.pi, 2*np.pi])
    ax2.set_ylim([-5*np.pi, 7*np.pi])
    # Legends
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    # Grid
    ax1.grid()
    ax2.grid()

    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved stacked theta plot to {pdf_path}")