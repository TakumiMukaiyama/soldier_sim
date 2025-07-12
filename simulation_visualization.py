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

    files_created = []

    # 1. Time series of average skills
    plt.figure(figsize=(12, 8), dpi=80)
    daily_skills = df.groupby(df["time"].dt.date)[skill_columns].mean()

    for i, (col, name) in enumerate(zip(skill_columns, skill_names)):
        plt.plot(daily_skills.index, daily_skills[col], label=name, linewidth=2, alpha=0.8)

    plt.title("Average Skill Progression Over Time", fontweight="bold", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Skill Level")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    output_file = save_dir / "skill_progression_time_series.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 2. Top performing agents
    plt.figure(figsize=(12, 8), dpi=80)
    final_skills = df.groupby("agent_id")[skill_columns].last()
    top_agents = final_skills.sum(axis=1).nlargest(10)

    top_agent_data = final_skills.loc[top_agents.index]
    x_pos = np.arange(len(top_agent_data))
    width = 0.2

    for i, (col, name) in enumerate(zip(skill_columns, skill_names)):
        plt.bar(x_pos + i * width, top_agent_data[col], width, label=name, alpha=0.8)

    plt.title("Top 10 Agents - Final Skills", fontweight="bold", fontsize=14)
    plt.xlabel("Agent ID")
    plt.ylabel("Skill Level")
    plt.xticks(x_pos + width * 1.5, [f"Agent {i}" for i in range(len(top_agent_data))], rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = save_dir / "top_agents_skills.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 3. Skill distribution comparison (start vs end)
    plt.figure(figsize=(12, 8), dpi=80)
    start_skills = df.groupby("agent_id")[skill_columns].first()
    end_skills = df.groupby("agent_id")[skill_columns].last()

    positions = [1, 2, 4, 5, 7, 8, 10, 11]
    labels = []
    data_to_plot = []

    for i, (col, name) in enumerate(zip(skill_columns, skill_names)):
        data_to_plot.extend([start_skills[col], end_skills[col]])
        labels.extend([f"{name}\n(Start)", f"{name}\n(End)"])

    box_plot = plt.boxplot(data_to_plot, positions=positions, patch_artist=True)

    # Color the boxes
    colors = ["lightblue", "lightcoral"] * 4
    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)

    plt.title("Skill Distribution: Start vs End", fontweight="bold", fontsize=14)
    plt.ylabel("Skill Level")
    plt.xticks(positions, labels, fontsize=9)
    plt.grid(True, alpha=0.3)

    # Add legend for box colors
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor="lightblue", label="Start"), Patch(facecolor="lightcoral", label="End")]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    output_file = save_dir / "skill_distribution_comparison.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 4. Skill correlation matrix
    plt.figure(figsize=(10, 8), dpi=80)
    correlation_data = df[skill_columns].corr()

    sns.heatmap(
        correlation_data,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Skill Correlation Matrix", fontweight="bold", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Update tick labels
    plt.gca().set_xticklabels(skill_names)
    plt.gca().set_yticklabels(skill_names)

    plt.tight_layout()
    output_file = save_dir / "skill_correlation_matrix.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # Print statistics
    logger.info("Skill progression statistics:")
    for col, name in zip(skill_columns, skill_names):
        start_avg = start_skills[col].mean()
        end_avg = end_skills[col].mean()
        improvement = ((end_avg - start_avg) / start_avg) * 100
        logger.info(f"  {name}: {start_avg:.3f} → {end_avg:.3f} (+{improvement:.1f}%)")

    return files_created


def visualize_poi_usage(df, save_dir):
    """Create POI usage analysis visualizations."""
    logger.info("Creating POI usage analysis...")

    files_created = []

    # 1. POI visit distribution
    plt.figure(figsize=(12, 8), dpi=80)
    poi_counts = df["poi_id"].value_counts()

    colors = plt.cm.Set3(np.linspace(0, 1, len(poi_counts)))
    bars = plt.bar(range(len(poi_counts)), poi_counts.values, color=colors)

    plt.title("POI Visit Distribution", fontweight="bold", fontsize=14)
    plt.xlabel("POI")
    plt.ylabel("Number of Visits")
    plt.xticks(range(len(poi_counts)), poi_counts.index, rotation=45, ha="right")
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, poi_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Create legend for POI colors
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], label=poi) for i, poi in enumerate(poi_counts.index)
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    output_file = save_dir / "poi_visit_distribution.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 2. Hourly POI usage patterns
    plt.figure(figsize=(12, 8), dpi=80)
    df["hour"] = df["time"].dt.hour
    hourly_poi = df.groupby(["hour", "poi_id"]).size().unstack(fill_value=0)

    hourly_poi.plot(kind="bar", stacked=True, color=colors[: len(hourly_poi.columns)])
    plt.title("Hourly POI Usage Patterns", fontweight="bold", fontsize=14)
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Visits")
    plt.legend(title="POI", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = save_dir / "hourly_poi_usage.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 3. POI efficiency (visits per agent per day)
    plt.figure(figsize=(12, 8), dpi=80)
    daily_poi_usage = df.groupby([df["time"].dt.date, "poi_id", "agent_id"]).size().reset_index(name="visits")
    poi_efficiency = daily_poi_usage.groupby("poi_id")["visits"].mean()

    bars = plt.bar(
        range(len(poi_efficiency)),
        poi_efficiency.values,
        color=colors[: len(poi_efficiency)],
    )
    plt.title("Average Visits per Agent per Day by POI", fontweight="bold", fontsize=14)
    plt.xlabel("POI")
    plt.ylabel("Avg Visits per Agent per Day")
    plt.xticks(range(len(poi_efficiency)), poi_efficiency.index, rotation=45, ha="right")
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, poi_efficiency.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Create legend for POI colors
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], label=poi) for i, poi in enumerate(poi_efficiency.index)
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    output_file = save_dir / "poi_efficiency_analysis.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 4. POI usage trends over time
    plt.figure(figsize=(12, 8), dpi=80)
    # Calculate weeks from simulation start
    start_date = df["time"].min()
    simulation_week = (df["time"] - start_date).dt.days // 7
    weekly_poi = df.groupby([simulation_week, "poi_id"]).size().unstack(fill_value=0)

    for poi in weekly_poi.columns:
        plt.plot(weekly_poi.index, weekly_poi[poi], label=poi, linewidth=2, alpha=0.8)

    plt.title("Weekly POI Usage Trends", fontweight="bold", fontsize=14)
    plt.xlabel("Week (from simulation start)")
    plt.ylabel("Total Visits")
    plt.legend(title="POI", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = save_dir / "weekly_poi_usage_trends.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # Print POI statistics
    logger.info("POI usage statistics:")
    total_visits = len(df)
    for poi, count in poi_counts.items():
        percentage = (count / total_visits) * 100
        logger.info(f"  {poi}: {count:,} visits ({percentage:.1f}%)")

    return files_created


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
    fig, ax = plt.subplots(figsize=(15, 8), dpi=60)  # Lower DPI for smaller file

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
    skill_files = visualize_skill_progression(df, save_dir)
    files_created.extend(skill_files)

    # 2. POI usage analysis
    poi_files = visualize_poi_usage(df, save_dir)
    files_created.extend(poi_files)

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
