#!/usr/bin/env python3
"""
Military Multi-Agent Simulation Visualization
Based on actual simulation data structure analysis
"""

import argparse
import ast
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation

from src.utils.get_logger import get_logger

# Initialize logger
logger = get_logger(__name__)

warnings.filterwarnings("ignore")


def load_and_process_data(log_dir):
    """Load and process simulation data from parquet and json files."""
    # Load temporal memory data
    output_dir = Path(log_dir)
    parquet_file = output_dir / "temporal_memory.parquet"
    logs_file = output_dir / "logs.json"

    logger.info(f"Loading data from {parquet_file}")
    df = pd.read_parquet(parquet_file)

    # Load logs data
    with open(logs_file, "r") as f:
        logs_data = json.load(f)

    # Convert to DataFrame
    logs_df = pd.DataFrame(logs_data)

    # Parse location coordinates from numpy array format
    def parse_location(location_array):
        try:
            if location_array is None:
                return None, None
            # Location is already a numpy array [x, y]
            if hasattr(location_array, "__len__") and len(location_array) == 2:
                return float(location_array[0]), float(location_array[1])
            else:
                return None, None
        except:
            return None, None

    # Parse locations
    locations = df["location"].apply(parse_location)
    df["x"] = [loc[0] if loc[0] is not None else np.nan for loc in locations]
    df["y"] = [loc[1] if loc[1] is not None else np.nan for loc in locations]

    # Parse observations to extract specific values
    def parse_observation(obs_str):
        if pd.isna(obs_str):
            return {}
        try:
            # Convert string representation to dictionary
            obs_dict = ast.literal_eval(obs_str)
            return obs_dict
        except:
            return {}

    # Parse observations
    df["parsed_observation"] = df["observation"].apply(parse_observation)

    # Extract specific observation fields
    observation_fields = [
        "energy",
        "social",
        "weapon_strength",
        "hunger",
        "management_skill",
        "sociability",
        "power",
    ]
    for field in observation_fields:
        df[f"obs_{field}"] = df["parsed_observation"].apply(lambda x: x.get(field, np.nan))

    # Convert time to datetime
    df["time"] = pd.to_datetime(df["time"])

    logger.info(f"Loaded {len(df)} records from temporal memory")
    logger.info(f"Loaded {len(logs_df)} records from logs")
    logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
    logger.info(f"Number of agents: {df['agent_id'].nunique()}")
    logger.info(f"Number of POIs: {df['poi_id'].nunique()}")

    return df, logs_df, output_dir


def visualize_skill_progression(df, save_dir):
    """Create skill progression visualizations."""
    logger.info("Creating skill progression analysis...")

    # Skill columns to analyze
    skill_columns = [
        "current_weapon_strength",
        "current_management_skill",
        "current_sociability",
        "current_power",
    ]
    skill_names = ["Weapon Strength", "Management Skill", "Sociability", "Power"]

    # Create figure with subplots (reduced DPI for smaller file size)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=80)
    fig.suptitle("Agent Skill Progression Analysis", fontsize=16, fontweight="bold")

    # 1. Time series of average skills
    ax1 = axes[0, 0]
    daily_skills = df.groupby(df["time"].dt.date)[skill_columns].mean()

    for i, (col, name) in enumerate(zip(skill_columns, skill_names)):
        ax1.plot(daily_skills.index, daily_skills[col], label=name, linewidth=2, alpha=0.8)

    ax1.set_title("Average Skill Progression Over Time", fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Skill Level")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # 2. Top performing agents
    ax2 = axes[0, 1]
    final_skills = df.groupby("agent_id")[skill_columns].last()
    top_agents = final_skills.sum(axis=1).nlargest(10)

    top_agent_data = final_skills.loc[top_agents.index]
    x_pos = np.arange(len(top_agent_data))
    width = 0.2  # Adjusted for 4 skills

    for i, (col, name) in enumerate(zip(skill_columns, skill_names)):
        ax2.bar(x_pos + i * width, top_agent_data[col], width, label=name, alpha=0.8)

    ax2.set_title("Top 10 Agents - Final Skills", fontweight="bold")
    ax2.set_xlabel("Agent ID")
    ax2.set_ylabel("Skill Level")
    ax2.set_xticks(x_pos + width * 1.5)  # Center the labels for 4 bars
    ax2.set_xticklabels([f"Agent {i}" for i in range(len(top_agent_data))], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Skill distribution comparison (start vs end)
    ax3 = axes[1, 0]
    start_skills = df.groupby("agent_id")[skill_columns].first()
    end_skills = df.groupby("agent_id")[skill_columns].last()

    positions = [1, 2, 4, 5, 7, 8, 10, 11]  # Extended for 4 skills
    labels = []
    data_to_plot = []

    for i, (col, name) in enumerate(zip(skill_columns, skill_names)):
        data_to_plot.extend([start_skills[col], end_skills[col]])
        labels.extend([f"{name}\n(Start)", f"{name}\n(End)"])

    box_plot = ax3.boxplot(data_to_plot, positions=positions, patch_artist=True)

    # Color the boxes
    colors = ["lightblue", "lightcoral"] * 4  # Extended for 4 skills
    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)

    ax3.set_title("Skill Distribution: Start vs End", fontweight="bold")
    ax3.set_ylabel("Skill Level")
    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Skill correlation matrix
    ax4 = axes[1, 1]
    correlation_data = df[skill_columns].corr()

    sns.heatmap(
        correlation_data,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax4,
        cbar_kws={"shrink": 0.8},
    )
    ax4.set_title("Skill Correlation Matrix", fontweight="bold")
    ax4.set_xticklabels(skill_names, rotation=45)
    ax4.set_yticklabels(skill_names, rotation=0)

    plt.tight_layout()

    # Save with smaller file size
    output_file = save_dir / "skill_progression_analysis.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()

    # Print statistics
    logger.info("Skill progression statistics:")
    for col, name in zip(skill_columns, skill_names):
        start_avg = start_skills[col].mean()
        end_avg = end_skills[col].mean()
        improvement = ((end_avg - start_avg) / start_avg) * 100
        logger.info(f"  {name}: {start_avg:.3f} → {end_avg:.3f} (+{improvement:.1f}%)")

    return output_file


def visualize_poi_usage(df, save_dir):
    """Create POI usage analysis visualizations."""
    logger.info("Creating POI usage analysis...")

    # Create figure with subplots (reduced DPI)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=80)
    fig.suptitle("POI Usage Analysis", fontsize=16, fontweight="bold")

    # 1. POI visit distribution
    ax1 = axes[0, 0]
    poi_counts = df["poi_id"].value_counts()

    colors = plt.cm.Set3(np.linspace(0, 1, len(poi_counts)))
    bars = ax1.bar(range(len(poi_counts)), poi_counts.values, color=colors)
    ax1.set_title("POI Visit Distribution", fontweight="bold")
    ax1.set_xlabel("POI")
    ax1.set_ylabel("Number of Visits")
    ax1.set_xticks(range(len(poi_counts)))
    ax1.set_xticklabels(poi_counts.index, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, poi_counts.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. Hourly POI usage patterns
    ax2 = axes[0, 1]
    df["hour"] = df["time"].dt.hour
    hourly_poi = df.groupby(["hour", "poi_id"]).size().unstack(fill_value=0)

    hourly_poi.plot(kind="bar", stacked=True, ax=ax2, color=colors[: len(hourly_poi.columns)])
    ax2.set_title("Hourly POI Usage Patterns", fontweight="bold")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Number of Visits")
    ax2.legend(title="POI", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    # 3. POI efficiency (visits per agent per day)
    ax3 = axes[1, 0]
    daily_poi_usage = df.groupby([df["time"].dt.date, "poi_id", "agent_id"]).size().reset_index(name="visits")
    poi_efficiency = daily_poi_usage.groupby("poi_id")["visits"].mean()

    bars = ax3.bar(
        range(len(poi_efficiency)),
        poi_efficiency.values,
        color=colors[: len(poi_efficiency)],
    )
    ax3.set_title("Average Visits per Agent per Day by POI", fontweight="bold")
    ax3.set_xlabel("POI")
    ax3.set_ylabel("Avg Visits per Agent per Day")
    ax3.set_xticks(range(len(poi_efficiency)))
    ax3.set_xticklabels(poi_efficiency.index, rotation=45, ha="right")
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, poi_efficiency.values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 4. POI usage trends over time
    ax4 = axes[1, 1]
    # Calculate weeks from simulation start
    start_date = df["time"].min()
    simulation_week = (df["time"] - start_date).dt.days // 7
    weekly_poi = df.groupby([simulation_week, "poi_id"]).size().unstack(fill_value=0)

    for poi in weekly_poi.columns:
        ax4.plot(weekly_poi.index, weekly_poi[poi], label=poi, linewidth=2, alpha=0.8)

    ax4.set_title("Weekly POI Usage Trends", fontweight="bold")
    ax4.set_xlabel("Week (from simulation start)")
    ax4.set_ylabel("Total Visits")
    ax4.legend(title="POI")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save with smaller file size
    output_file = save_dir / "poi_usage_analysis.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()

    # Print POI statistics
    logger.info("POI usage statistics:")
    total_visits = len(df)
    for poi, count in poi_counts.items():
        percentage = (count / total_visits) * 100
        logger.info(f"  {poi}: {count:,} visits ({percentage:.1f}%)")

    return output_file


def create_agent_movement_animation(df, save_dir):
    """Create animated visualization of agent movements."""
    logger.info("Creating agent movement animation...")

    # Filter data with valid coordinates
    df_with_coords = df.dropna(subset=["x", "y"])

    if len(df_with_coords) == 0:
        logger.warning("No valid coordinates found for animation")
        return None

    # Sample data for animation (every 6 hours to reduce file size)
    df_sampled = df_with_coords[df_with_coords["time"].dt.hour.isin([0, 6, 12, 18])]

    # Get unique time points (limit to first 24 to keep file size manageable)
    unique_times = sorted(df_sampled["time"].unique())[:24]

    # POI locations (approximated from data)
    poi_locations = df_sampled.groupby("poi_id")[["x", "y"]].mean()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=60)  # Lower DPI for smaller file

    def animate(frame):
        ax.clear()

        current_time = unique_times[frame]
        current_data = df_sampled[df_sampled["time"] == current_time]

        # Plot POI locations
        for poi, location in poi_locations.iterrows():
            ax.scatter(location["x"], location["y"], s=200, alpha=0.7, label=poi, marker="s")

        # Plot agent locations
        if len(current_data) > 0:
            ax.scatter(
                current_data["x"],
                current_data["y"],
                c="red",
                s=30,
                alpha=0.6,
                label="Agents",
            )

        ax.set_title(
            f"Agent Movement - {current_time.strftime('%Y-%m-%d %H:%M')}",
            fontweight="bold",
        )
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Set consistent axis limits
        ax.set_xlim(df_with_coords["x"].min() - 1, df_with_coords["x"].max() + 1)
        ax.set_ylim(df_with_coords["y"].min() - 1, df_with_coords["y"].max() + 1)

    # Create animation with reduced parameters for smaller file size
    anim = FuncAnimation(fig, animate, frames=len(unique_times), interval=500, repeat=True, blit=False)

    # Save as GIF with optimization
    output_file = save_dir / "agent_movement.gif"
    anim.save(output_file, writer="pillow", fps=2)
    plt.close()

    logger.info(f"Animation saved with {len(unique_times)} frames")
    return output_file


def main():
    """Main function to run all visualizations."""
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Generate simulation visualizations for tech blog")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="output/20250711_235023",
        help="Directory containing logs.json and temporal_memory.parquet files (default: output/20250711_235023)",
    )

    args = parser.parse_args()

    logger.info("Starting simulation visualization generation...")
    logger.info(f"Using log directory: {args.log_dir}")

    # Load data
    df, logs_df, save_dir = load_and_process_data(args.log_dir)

    # Create visualizations
    files_created = []

    # 1. Skill progression analysis
    skill_file = visualize_skill_progression(df, save_dir)
    files_created.append(skill_file)

    # 2. POI usage analysis
    poi_file = visualize_poi_usage(df, save_dir)
    files_created.append(poi_file)

    # 3. Agent movement animation (GIF only)
    movement_file = create_agent_movement_animation(df, save_dir)
    if movement_file:
        files_created.append(movement_file)

    logger.info("\n" + "=" * 50)
    logger.info("VISUALIZATION GENERATION COMPLETE")
    logger.info("=" * 50)

    logger.info(f"\nFiles created in {save_dir}:")
    for file_path in files_created:
        if file_path and file_path.exists():
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            logger.info(f"  • {file_path.name} ({file_size:.1f} MB)")
        else:
            logger.warning(f"  • {file_path} (Failed to create)")

    logger.info("\nVisualization files are ready for tech blog use!")
    logger.info("All PNG files are optimized to be under 3MB for upload.")


if __name__ == "__main__":
    main()
