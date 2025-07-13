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
        except Exception as e:
            logger.error(f"Error parsing location: {e}")
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
            # Check if it's already a dictionary
            if isinstance(obs_str, dict):
                return obs_str
            # Convert string representation to dictionary
            obs_dict = ast.literal_eval(obs_str)
            return obs_dict
        except Exception as e:
            logger.debug(f"Error parsing observation: {e}")  # Changed to debug level
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
        df[f"obs_{field}"] = df["parsed_observation"].apply(
            lambda x: x.get(field, np.nan)
        )

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
        plt.plot(
            daily_skills.index, daily_skills[col], label=name, linewidth=2, alpha=0.8
        )

    plt.title("Average Skill Progression Over Time", fontweight="bold", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Skill Level")
    plt.ylim(30, 100)  # Set y-axis range from 30 to 100
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
    plt.xticks(
        x_pos + width * 1.5,
        [f"Agent {i}" for i in range(len(top_agent_data))],
        rotation=45,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = save_dir / "top_agents_skills.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 3. Skill distribution histograms
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=80)
    axes = axes.flatten()

    for i, (col, name) in enumerate(zip(skill_columns, skill_names)):
        final_values = df.groupby("agent_id")[col].last()
        axes[i].hist(final_values, bins=20, alpha=0.7, edgecolor="black")
        axes[i].set_title(f"{name} Distribution", fontweight="bold", fontsize=12)
        axes[i].set_xlabel("Skill Level", fontsize=11)
        axes[i].set_ylabel("Number of Agents", fontsize=11)
        axes[i].grid(True, alpha=0.3)

        # Rotate x-axis labels to prevent overlap
        axes[i].tick_params(axis="x", rotation=45, labelsize=10)
        axes[i].tick_params(axis="y", labelsize=10)

    plt.suptitle("Final Skill Distributions", fontweight="bold", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    output_file = save_dir / "skill_distributions.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 4. Skill correlation matrix
    plt.figure(figsize=(10, 8), dpi=80)
    final_skills = df.groupby("agent_id")[skill_columns].last()
    correlation = final_skills.corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        xticklabels=skill_names,
        yticklabels=skill_names,
    )
    plt.title("Skill Correlation Matrix", fontweight="bold", fontsize=14)
    plt.tight_layout()
    output_file = save_dir / "skill_correlation_matrix.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # Print statistics
    logger.info("Skill progression statistics:")
    for col, name in zip(skill_columns, skill_names):
        initial_avg = df.groupby("agent_id")[col].first().mean()
        final_avg = df.groupby("agent_id")[col].last().mean()
        growth = final_avg - initial_avg
        logger.info(f"  {name}: {initial_avg:.1f} -> {final_avg:.1f} (+{growth:.1f})")

    return files_created


def load_poi_categories():
    """Load POI categories mapping from pois.json"""
    try:
        from src.utils.df_utils import load_poi_categories as utils_load_poi_categories

        return utils_load_poi_categories()
    except ImportError:
        # Fallback to local implementation if the utility function is unavailable
        try:
            import json
            from pathlib import Path

            poi_file = Path("data/pois.json")
            if not poi_file.exists():
                return {}

            with open(poi_file, "r", encoding="utf-8") as f:
                pois = json.load(f)

            # Create mapping from POI ID to category
            poi_categories = {}
            for poi in pois:
                poi_categories[poi["id"]] = poi["category"]

            return poi_categories
        except Exception as e:
            logger.warning(f"Failed to load POI categories: {e}")
            return {}


def aggregate_poi_by_category_pandas(df):
    """Aggregate POI visits by category instead of individual POI IDs using pandas"""
    # Load POI categories
    poi_categories = load_poi_categories()

    # Add category column
    df = df.copy()
    df["poi_category"] = df["poi_id"].map(lambda x: poi_categories.get(x, "unknown"))

    # Replace poi_id with poi_category for aggregation
    df_agg = df.copy()
    df_agg["poi_id"] = df_agg["poi_category"]

    return df_agg


def setup_plot_style(title, xlabel, ylabel, legend_title=None, figsize=(12, 8)):
    """Setup common plot style parameters"""
    plt.figure(figsize=figsize, dpi=80)
    plt.title(title, fontweight="bold", fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    return plt.gca()  # Return current axes


def get_hour_interval_labels():
    """Get hour interval labels for 3-hour groups"""
    return [f"{h:02d}:00-{h + 2:02d}:59" for h in [0, 3, 6, 9, 12, 15, 18, 21]]


def prepare_poi_usage_data(df, aggregate_by_category=False):
    """Prepare data for POI usage analysis

    Args:
        df: DataFrame with POI visit data
        aggregate_by_category: If True, aggregate POIs by category

    Returns:
        Tuple of (processed_df, poi_column, all_hour_groups, title_text, y_label, csv_suffix)
    """
    # Define all possible POI categories from archetype definitions
    ALL_POI_CATEGORIES = [
        "training",
        "armory",
        "food",
        "rest",
        "sleep",
        "office",
        "recreation",
        "medical",
        "fitness",
        "library",
        "workshop",
        "communications",
        "maintenance",
        "outdoor",
        "spiritual",
        "logistics",
    ]

    # Extract hour from time and group into 3-hour intervals
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["hour_group"] = (df["hour"] // 3) * 3  # Map hours to 3-hour intervals

    # Aggregate by category if requested
    if aggregate_by_category:
        df = aggregate_poi_by_category_pandas(df)
        poi_column = "poi_category"
        y_label = "POI Category"
        title_text = "POI Category Usage Heatmap by 3-Hour Intervals"
        csv_suffix = "_by_category"

        # Ensure all POI categories are present
        for category in ALL_POI_CATEGORIES:
            if not any(df[poi_column] == category):
                # Add dummy row for missing categories
                dummy_row = df.iloc[0].copy()
                dummy_row[poi_column] = category
                dummy_row["hour_group"] = 0
                df = pd.concat([df, dummy_row.to_frame().T], ignore_index=True)
    else:
        poi_column = "poi_id"
        y_label = "POI ID"
        title_text = "POI Usage Heatmap by 3-Hour Intervals"
        csv_suffix = ""

    all_hour_groups = [0, 3, 6, 9, 12, 15, 18, 21]
    return df, poi_column, all_hour_groups, title_text, y_label, csv_suffix


def create_poi_usage_heatmap_table(df, save_dir, aggregate_by_category=False):
    """Create POI usage heatmap table with optional category aggregation.

    Args:
        df: DataFrame with POI visit data
        save_dir: Directory to save output files
        aggregate_by_category: If True, aggregate POIs by category
    """
    logger.info("Creating POI usage heatmap table...")

    # Prepare data
    df, poi_column, all_hour_groups, title_text, y_label, csv_suffix = (
        prepare_poi_usage_data(df, aggregate_by_category)
    )

    # Group by POI (or category) and hour group
    poi_hourly = df.groupby([poi_column, "hour_group"]).size().unstack(fill_value=0)

    if aggregate_by_category:
        # Sort by category name
        poi_hourly = poi_hourly.reindex(sorted(poi_hourly.index))

    # Ensure all 3-hour intervals are represented
    for hour_group in all_hour_groups:
        if hour_group not in poi_hourly.columns:
            poi_hourly[hour_group] = 0
    poi_hourly = poi_hourly.reindex(columns=sorted(poi_hourly.columns))

    # Create heatmap
    plt.figure(figsize=(12, max(12, len(poi_hourly) * 0.4)))

    # Create heatmap with seaborn
    sns.heatmap(
        poi_hourly,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "Number of Visits"},
        linewidths=0.5,
        square=False,
    )

    plt.title(title_text, fontweight="bold", fontsize=16)
    plt.xlabel("Time of Day (3-hour intervals)", fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Set x-axis labels for 3-hour intervals
    hour_labels = get_hour_interval_labels()
    plt.xticks(range(len(all_hour_groups)), hour_labels, rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    output_file = save_dir / f"poi_usage_heatmap_table{csv_suffix}.png"
    plt.savefig(output_file, dpi=100, bbox_inches="tight", format="png")
    plt.close()

    # Also create a summary table as CSV
    poi_summary = (
        df.groupby(poi_column)
        .agg({poi_column: "count", "hour_group": ["min", "max", "mean"]})
        .round(2)
    )
    poi_summary.columns = [
        "Total_Visits",
        "First_Hour_Group",
        "Last_Hour_Group",
        "Avg_Hour_Group",
    ]
    poi_summary = poi_summary.sort_values("Total_Visits", ascending=False)

    csv_file = save_dir / f"poi_usage_summary{csv_suffix}.csv"
    poi_summary.to_csv(csv_file)

    logger.info(
        f"Created POI usage heatmap table for {len(poi_hourly)} {'categories' if aggregate_by_category else 'POIs'}"
    )
    logger.info(f"POI usage summary saved to {csv_file}")

    return [output_file, csv_file]


def load_poi_metadata(log_dir):
    """Load POI metadata from JSON file."""
    try:
        import json
        from pathlib import Path

        # Try to find POI metadata file
        poi_files = ["pois_improved.json", "pois.json"]
        pois_metadata = {}

        for poi_file in poi_files:
            poi_path = Path("data") / poi_file
            if poi_path.exists():
                with open(poi_path, "r") as f:
                    pois_data = json.load(f)
                    pois_metadata = {poi["id"]: poi for poi in pois_data}
                logger.info(
                    f"Loaded POI metadata from {poi_path} ({len(pois_metadata)} POIs)"
                )
                break
        else:
            logger.warning("No POI metadata file found, using POI IDs only")

        return pois_metadata
    except Exception as e:
        logger.warning(f"Failed to load POI metadata: {e}")
        return {}


def visualize_poi_usage(df, save_dir, create_both_versions=True):
    """Create POI usage analysis visualizations.

    Args:
        df: DataFrame with POI visit data
        save_dir: Directory to save output files
        create_both_versions: If True, create both individual POI and category-aggregated versions
    """
    logger.info("Creating POI usage analysis...")

    files_created = []

    # Create consistent color mappings for POIs and categories
    poi_categories = load_poi_categories()
    unique_pois = sorted(df["poi_id"].unique())
    unique_categories = sorted(list(set(poi_categories.values())))

    # Create consistent color mappings
    poi_colors = {}
    poi_color_palette = plt.cm.Set3(np.linspace(0, 1, len(unique_pois)))
    for i, poi in enumerate(unique_pois):
        poi_colors[poi] = poi_color_palette[i]

    category_colors = {}
    category_color_palette = plt.cm.Set2(np.linspace(0, 1, len(unique_categories)))
    for i, category in enumerate(unique_categories):
        category_colors[category] = category_color_palette[i]

    # Create individual POI heatmap
    heatmap_files_individual = create_poi_usage_heatmap_table(
        df, save_dir, aggregate_by_category=False
    )
    files_created.extend(heatmap_files_individual)

    # Create category-aggregated heatmap if requested
    if create_both_versions:
        heatmap_files_category = create_poi_usage_heatmap_table(
            df, save_dir, aggregate_by_category=True
        )
        files_created.extend(heatmap_files_category)

    # 1. POI visit distribution
    plt.figure(figsize=(12, 8), dpi=80)
    poi_counts = df["poi_id"].value_counts()

    # Use consistent colors for POIs
    colors = [poi_colors[poi] for poi in poi_counts.index]
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

    # Create legend for POI colors - show all POIs
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=poi_colors[poi], label=poi)
        for poi in poi_counts.index
    ]
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="All POIs",
        fontsize=8,
    )

    plt.tight_layout()
    output_file = save_dir / "poi_visit_distribution.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 2. Improved hourly POI usage patterns with better legend
    plt.figure(figsize=(12, 10), dpi=80)
    df["hour"] = df["time"].dt.hour

    # Group hours into 3-hour intervals
    df["hour_group"] = (df["hour"] // 3) * 3
    hourly_poi = df.groupby(["hour_group", "poi_id"]).size().unstack(fill_value=0)

    # Create stacked bar chart with consistent colors
    poi_colors_ordered = [poi_colors[poi] for poi in hourly_poi.columns]
    _ = hourly_poi.plot(
        kind="bar", stacked=True, figsize=(12, 10), color=poi_colors_ordered
    )

    plt.title("POI Usage Patterns by 3-Hour Intervals", fontweight="bold", fontsize=16)
    plt.xlabel("Time of Day (3-hour intervals)", fontsize=12)
    plt.ylabel("Number of Visits", fontsize=12)

    # Set x-axis labels for 3-hour intervals
    hour_labels = [f"{h:02d}:00-{h + 2:02d}:59" for h in [0, 3, 6, 9, 12, 15, 18, 21]]
    plt.xticks(range(len([0, 3, 6, 9, 12, 15, 18, 21])), hour_labels, rotation=45)
    plt.grid(True, alpha=0.3)

    # Improve legend - show all POIs
    plt.legend(
        title="POI ID",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=10,
    )

    plt.tight_layout()
    output_file = save_dir / "hourly_poi_usage.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 3. POI efficiency (visits per agent per day) - Individual POIs
    plt.figure(figsize=(12, 8), dpi=80)
    daily_poi_usage = (
        df.groupby([df["time"].dt.date, "poi_id", "agent_id"])
        .size()
        .reset_index(name="visits")
    )
    poi_efficiency = daily_poi_usage.groupby("poi_id")["visits"].mean()

    # Use consistent colors for POIs
    efficiency_colors = [poi_colors[poi] for poi in poi_efficiency.index]
    bars = plt.bar(
        range(len(poi_efficiency)),
        poi_efficiency.values,
        color=efficiency_colors,
    )
    plt.title("Average Visits per Agent per Day by POI", fontweight="bold", fontsize=14)
    plt.xlabel("POI")
    plt.ylabel("Avg Visits per Agent per Day")
    plt.xticks(
        range(len(poi_efficiency)), poi_efficiency.index, rotation=45, ha="right"
    )
    plt.ylim(0, 5)  # Set y-axis maximum to 5
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

    # Create legend for POI colors - show all POIs
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=poi_colors[poi], label=poi)
        for poi in poi_efficiency.index
    ]
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="All POIs",
        fontsize=8,
    )

    plt.tight_layout()
    output_file = save_dir / "poi_efficiency_analysis.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 3b. POI efficiency by category (if requested)
    if create_both_versions:
        plt.figure(figsize=(12, 8), dpi=80)

        # Create category-aggregated data
        df_category = aggregate_poi_by_category_pandas(df)
        daily_category_usage = (
            df_category.groupby([df_category["time"].dt.date, "poi_id", "agent_id"])
            .size()
            .reset_index(name="visits")
        )
        category_efficiency = daily_category_usage.groupby("poi_id")["visits"].mean()

        # Use consistent colors for categories
        efficiency_category_colors = [
            category_colors[category] for category in category_efficiency.index
        ]
        bars = plt.bar(
            range(len(category_efficiency)),
            category_efficiency.values,
            color=efficiency_category_colors,
        )
        plt.title(
            "Average Visits per Agent per Day by POI Category",
            fontweight="bold",
            fontsize=14,
        )
        plt.xlabel("POI Category")
        plt.ylabel("Avg Visits per Agent per Day")
        plt.xticks(
            range(len(category_efficiency)),
            category_efficiency.index,
            rotation=45,
            ha="right",
        )
        plt.ylim(0, 5)  # Set y-axis maximum to 5
        plt.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, category_efficiency.values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Create legend for category colors
        legend_elements = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=category_colors[category], label=category
            )
            for category in category_efficiency.index
        ]
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title="POI Categories",
            fontsize=8,
        )

        plt.tight_layout()
        output_file = save_dir / "poi_efficiency_analysis_by_category.png"
        plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
        plt.close()
        files_created.append(output_file)

    # 4. POI usage trends over time - Individual POIs
    plt.figure(figsize=(12, 8), dpi=80)
    # Calculate weeks from simulation start
    start_date = df["time"].min()
    simulation_week = (df["time"] - start_date).dt.days // 7
    weekly_poi = df.groupby([simulation_week, "poi_id"]).size().unstack(fill_value=0)

    # Use consistent colors for POIs
    for poi in weekly_poi.columns:
        plt.plot(
            weekly_poi.index,
            weekly_poi[poi],
            label=poi,
            linewidth=2,
            alpha=0.8,
            color=poi_colors[poi],
        )

    plt.title("Weekly POI Usage Trends", fontweight="bold", fontsize=14)
    plt.xlabel("Week (from simulation start)")
    plt.ylabel("Total Visits")
    plt.legend(
        title="POI ID",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=10,
    )
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = save_dir / "weekly_poi_usage_trends.png"
    plt.savefig(output_file, dpi=80, bbox_inches="tight", format="png")
    plt.close()
    files_created.append(output_file)

    # 4b. POI usage trends over time by category (if requested)
    if create_both_versions:
        plt.figure(figsize=(12, 8), dpi=80)

        # Create category-aggregated data for weekly trends
        df_category_trends = aggregate_poi_by_category_pandas(df)
        weekly_category = (
            df_category_trends.groupby([simulation_week, "poi_id"])
            .size()
            .unstack(fill_value=0)
        )

        # Use consistent colors for categories
        for category in weekly_category.columns:
            plt.plot(
                weekly_category.index,
                weekly_category[category],
                label=category,
                linewidth=2,
                alpha=0.8,
                color=category_colors[category],
            )

        plt.title("Weekly POI Usage Trends by Category", fontweight="bold", fontsize=14)
        plt.xlabel("Week (from simulation start)")
        plt.ylabel("Total Visits")
        plt.legend(
            title="POI Category",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
            title_fontsize=10,
        )
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = save_dir / "weekly_poi_usage_trends_by_category.png"
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

    # Create consistent color mapping for POIs
    unique_pois = sorted(df_sampled["poi_id"].unique())
    poi_colors = {}
    poi_color_palette = plt.cm.Set3(np.linspace(0, 1, len(unique_pois)))
    for i, poi in enumerate(unique_pois):
        poi_colors[poi] = poi_color_palette[i]

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8), dpi=60)

    def animate(frame):
        ax.clear()

        current_time = unique_times[frame]
        current_data = df_sampled[df_sampled["time"] == current_time]

        # Plot POI locations with consistent colors
        for poi, location in poi_locations.iterrows():
            ax.scatter(
                location["x"],
                location["y"],
                s=200,
                alpha=0.7,
                label=poi,
                marker="s",
                color=poi_colors[poi],
            )

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
    anim = FuncAnimation(
        fig, animate, frames=len(unique_times), interval=500, repeat=True, blit=False
    )

    # Save as GIF with optimization
    output_file = save_dir / "agent_movement.gif"
    anim.save(output_file, writer="pillow", fps=2)
    plt.close()

    logger.info(f"Animation saved with {len(unique_times)} frames")
    return output_file


def main():
    """Main function to run all visualizations."""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Generate simulation visualizations for tech blog"
    )
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

    # 2. POI usage analysis (including new heatmap table)
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
    logger.info("\nNew POI usage heatmap table shows detailed hourly patterns!")


if __name__ == "__main__":
    main()
