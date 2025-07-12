import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import polars as pl


def append_records(df: pl.DataFrame, records: List[Dict]) -> pl.DataFrame:
    """Append multiple records to a DataFrame"""
    new_df = pl.DataFrame(records)
    return pl.concat([df, new_df])


def aggregate_daily_energy(memory_df: pl.DataFrame, agent_id: Optional[str] = None) -> pl.DataFrame:
    """
    Aggregate energy statistics per day

    Args:
        memory_df: Temporal memory DataFrame
        agent_id: Optional agent ID to filter by

    Returns:
        DataFrame with daily energy stats
    """
    filtered_df = memory_df
    if agent_id:
        filtered_df = filtered_df.filter(pl.col("agent_id") == agent_id)

    # Extract date from time
    filtered_df = filtered_df.with_columns(pl.col("time").str.slice(0, 10).alias("date"))

    # Group by date and get energy stats
    result = filtered_df.groupby("date").agg(
        pl.min("current_energy").alias("min_energy"),
        pl.mean("current_energy").alias("avg_energy"),
        pl.max("current_energy").alias("max_energy"),
        pl.count("current_energy").alias("actions"),
    )

    return result.sort("date")


def aggregate_daily_stats(memory_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate daily statistics for all agents

    Returns:
        DataFrame with daily agent statistics
    """
    # Extract date from time column
    df = memory_df.with_columns(
        pl.col("time").str.slice(0, 10).alias("date")  # Extract YYYY-MM-DD
    )

    # Group by date and agent_id, then aggregate
    daily_stats = df.groupby(["date", "agent_id"]).agg(
        [
            pl.col("current_energy").mean().alias("avg_energy"),
            pl.col("current_energy").max().alias("max_energy"),
            pl.col("current_energy").min().alias("min_energy"),
            pl.col("current_social").mean().alias("avg_social"),
            pl.col("current_weapon_strength").mean().alias("avg_weapon_strength"),
            pl.col("current_hunger").mean().alias("avg_hunger"),
            pl.col("current_management_skill").mean().alias("avg_management_skill"),
            pl.col("current_sociability").mean().alias("avg_sociability"),
            pl.col("current_power").mean().alias("avg_power"),
            pl.col("poi_id").count().alias("total_actions"),
        ]
    )

    return daily_stats.sort(["date", "agent_id"])


def poi_visit_stats(memory_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute statistics about POI visits

    Returns:
        DataFrame with POI visit counts and agent counts
    """
    return (
        memory_df.filter(pl.col("poi_id").is_not_null())
        .groupby("poi_id")
        .agg(
            pl.count().alias("visit_count"),
            pl.n_unique("agent_id").alias("unique_agents"),
        )
        .sort("visit_count", descending=True)
    )


def load_poi_categories() -> Dict[str, str]:
    """Load POI categories mapping from pois.json"""
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


def aggregate_poi_by_category(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate POI visits by category instead of individual POI IDs

    Args:
        df: DataFrame with POI visit data containing 'poi_id' column

    Returns:
        DataFrame with POI categories aggregated
    """
    # Load POI categories
    poi_categories = load_poi_categories()

    # Add category column
    df = df.with_columns(
        [
            pl.col("poi_id")
            .map_elements(lambda x: poi_categories.get(x, "unknown"), return_dtype=pl.Utf8)
            .alias("poi_category")
        ]
    )

    # Group by category and aggregate
    numeric_cols = [col for col in df.columns if col not in ["poi_id", "poi_category", "agent_id", "time"]]

    agg_exprs = []
    for col in numeric_cols:
        if df[col].dtype.is_numeric():
            agg_exprs.append(pl.col(col).sum())
        else:
            agg_exprs.append(pl.col(col).first())

    # Group by all non-numeric columns except poi_id
    group_cols = [col for col in df.columns if col not in ["poi_id"] + numeric_cols]
    if "poi_category" not in group_cols:
        group_cols.append("poi_category")

    if agg_exprs:
        aggregated = df.group_by(group_cols).agg(agg_exprs)
    else:
        aggregated = df.group_by(group_cols).agg(pl.len().alias("count"))

    return aggregated


def create_poi_usage_heatmap_table_polars(
    df: pl.DataFrame, aggregate_by_category: bool = False, width: int = 1000, height: int = 600
) -> go.Figure:
    """Create interactive POI usage heatmap table using Polars and Plotly

    Args:
        df: DataFrame with columns ['time', 'agent_id', 'poi_id', 'activity']
        aggregate_by_category: If True, aggregate POIs by category
        width: Figure width
        height: Figure height

    Returns:
        Plotly figure with interactive heatmap
    """
    # Convert time to hour groups (3-hour intervals)
    df = df.with_columns([pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").dt.hour().alias("hour")])

    df = df.with_columns([((pl.col("hour") // 3) * 3).alias("hour_group")])

    # Aggregate by category if requested
    if aggregate_by_category:
        df = aggregate_poi_by_category(df)
        poi_column = "poi_category"
        title_suffix = " (by Category)"
    else:
        poi_column = "poi_id"
        title_suffix = ""

    # Create hour group labels
    hour_labels = {
        0: "00:00-02:59",
        3: "03:00-05:59",
        6: "06:00-08:59",
        9: "09:00-11:59",
        12: "12:00-14:59",
        15: "15:00-17:59",
        18: "18:00-20:59",
        21: "21:00-23:59",
    }

    # Count visits by POI and hour group
    visit_counts = df.group_by([poi_column, "hour_group"]).agg([pl.len().alias("visit_count")])

    # Create pivot table
    pivot_data = visit_counts.pivot(index=poi_column, columns="hour_group", values="visit_count").fill_null(0)

    # Get all POIs and hour groups
    all_pois = sorted(pivot_data[poi_column].to_list())
    all_hours = sorted([0, 3, 6, 9, 12, 15, 18, 21])

    # Create matrix for heatmap
    matrix = []
    for poi in all_pois:
        row = []
        for hour in all_hours:
            hour_col = str(hour)
            if hour_col in pivot_data.columns:
                poi_data = pivot_data.filter(pl.col(poi_column) == poi)
                if len(poi_data) > 0:
                    value = poi_data[hour_col].to_list()[0]
                    row.append(value)
                else:
                    row.append(0)
            else:
                row.append(0)
        matrix.append(row)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[hour_labels[h] for h in all_hours],
            y=all_pois,
            colorscale="Viridis",
            showscale=True,
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Visit Count", titleside="right"),
        )
    )

    fig.update_layout(
        title=f"POI Usage Heatmap by 3-Hour Intervals{title_suffix}",
        xaxis_title="Time Period",
        yaxis_title="POI" if not aggregate_by_category else "POI Category",
        width=width,
        height=height,
        font=dict(size=12),
    )

    return fig


def create_agent_activity_heatmap(
    df: pl.DataFrame, aggregate_by_category: bool = False, width: int = 1000, height: int = 600
) -> go.Figure:
    """Create agent activity heatmap showing activity patterns

    Args:
        df: DataFrame with columns ['time', 'agent_id', 'poi_id', 'activity']
        aggregate_by_category: If True, aggregate POIs by category
        width: Figure width
        height: Figure height

    Returns:
        Plotly figure with agent activity heatmap
    """
    # Convert time to hour groups
    df = df.with_columns([pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").dt.hour().alias("hour")])

    df = df.with_columns([((pl.col("hour") // 3) * 3).alias("hour_group")])

    # Aggregate by category if requested
    if aggregate_by_category:
        df = aggregate_poi_by_category(df)
        poi_column = "poi_category"
        title_suffix = " (by Category)"
    else:
        poi_column = "poi_id"
        title_suffix = ""

    # Create hour group labels
    hour_labels = {
        0: "00:00-02:59",
        3: "03:00-05:59",
        6: "06:00-08:59",
        9: "09:00-11:59",
        12: "12:00-14:59",
        15: "15:00-17:59",
        18: "18:00-20:59",
        21: "21:00-23:59",
    }

    # Count activities by agent and hour group
    activity_counts = df.group_by(["agent_id", "hour_group"]).agg([pl.len().alias("activity_count")])

    # Create pivot table
    pivot_data = activity_counts.pivot(index="agent_id", columns="hour_group", values="activity_count").fill_null(0)

    # Get all agents and hour groups
    all_agents = sorted(pivot_data["agent_id"].to_list())
    all_hours = sorted([0, 3, 6, 9, 12, 15, 18, 21])

    # Create matrix for heatmap
    matrix = []
    for agent in all_agents:
        row = []
        for hour in all_hours:
            hour_col = str(hour)
            if hour_col in pivot_data.columns:
                agent_data = pivot_data.filter(pl.col("agent_id") == agent)
                if len(agent_data) > 0:
                    value = agent_data[hour_col].to_list()[0]
                    row.append(value)
                else:
                    row.append(0)
            else:
                row.append(0)
        matrix.append(row)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[hour_labels[h] for h in all_hours],
            y=all_agents,
            colorscale="Plasma",
            showscale=True,
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Activity Count", titleside="right"),
        )
    )

    fig.update_layout(
        title=f"Agent Activity Heatmap by 3-Hour Intervals{title_suffix}",
        xaxis_title="Time Period",
        yaxis_title="Agent ID",
        width=width,
        height=height,
        font=dict(size=12),
    )

    return fig


def analyze_peak_usage_times(df: pl.DataFrame, aggregate_by_category: bool = False) -> pl.DataFrame:
    """Analyze peak usage times for each POI

    Args:
        df: DataFrame with columns ['time', 'agent_id', 'poi_id', 'activity']
        aggregate_by_category: If True, aggregate POIs by category

    Returns:
        DataFrame with peak usage analysis
    """
    # Convert time to hour groups
    df = df.with_columns([pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").dt.hour().alias("hour")])

    df = df.with_columns([((pl.col("hour") // 3) * 3).alias("hour_group")])

    # Aggregate by category if requested
    if aggregate_by_category:
        df = aggregate_poi_by_category(df)
        poi_column = "poi_category"
    else:
        poi_column = "poi_id"

    # Create hour group labels
    hour_labels = {
        0: "00:00-02:59",
        3: "03:00-05:59",
        6: "06:00-08:59",
        9: "09:00-11:59",
        12: "12:00-14:59",
        15: "15:00-17:59",
        18: "18:00-20:59",
        21: "21:00-23:59",
    }

    # Count visits by POI and hour group
    visit_counts = df.group_by([poi_column, "hour_group"]).agg([pl.len().alias("visit_count")])

    # Find peak usage time for each POI
    peak_times = visit_counts.group_by(poi_column).agg(
        [
            pl.col("visit_count").max().alias("max_visits"),
            pl.col("hour_group")
            .filter(pl.col("visit_count") == pl.col("visit_count").max())
            .first()
            .alias("peak_hour_group"),
        ]
    )

    # Add hour group labels
    peak_times = peak_times.with_columns(
        [
            pl.col("peak_hour_group")
            .map_elements(lambda x: hour_labels.get(x, "Unknown"), return_dtype=pl.Utf8)
            .alias("peak_time_label")
        ]
    )

    # Calculate total visits for each POI
    total_visits = df.group_by(poi_column).agg([pl.len().alias("total_visits")])

    # Join with peak times
    result = peak_times.join(total_visits, on=poi_column)

    # Sort by total visits descending
    result = result.sort("total_visits", descending=True)

    return result


def create_activity_summary_table(df: pl.DataFrame, aggregate_by_category: bool = False) -> pl.DataFrame:
    """Create comprehensive activity summary table

    Args:
        df: DataFrame with columns ['time', 'agent_id', 'poi_id', 'activity']
        aggregate_by_category: If True, aggregate POIs by category

    Returns:
        DataFrame with activity summary statistics
    """
    # Convert time to hour groups
    df = df.with_columns([pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").dt.hour().alias("hour")])

    df = df.with_columns([((pl.col("hour") // 3) * 3).alias("hour_group")])

    # Aggregate by category if requested
    if aggregate_by_category:
        df = aggregate_poi_by_category(df)
        poi_column = "poi_category"
    else:
        poi_column = "poi_id"

    # Calculate summary statistics
    summary = df.group_by(poi_column).agg(
        [
            pl.len().alias("total_visits"),
            pl.col("agent_id").n_unique().alias("unique_agents"),
            pl.col("hour_group").n_unique().alias("active_time_periods"),
            pl.col("hour_group").mode().first().alias("most_common_hour_group"),
        ]
    )

    # Calculate average visits per agent
    summary = summary.with_columns(
        [(pl.col("total_visits") / pl.col("unique_agents")).round(2).alias("avg_visits_per_agent")]
    )

    # Create hour group labels
    hour_labels = {
        0: "00:00-02:59",
        3: "03:00-05:59",
        6: "06:00-08:59",
        9: "09:00-11:59",
        12: "12:00-14:59",
        15: "15:00-17:59",
        18: "18:00-20:59",
        21: "21:00-23:59",
    }

    # Add hour group labels
    summary = summary.with_columns(
        [
            pl.col("most_common_hour_group")
            .map_elements(lambda x: hour_labels.get(x, "Unknown"), return_dtype=pl.Utf8)
            .alias("most_common_time_period")
        ]
    )

    # Sort by total visits descending
    summary = summary.sort("total_visits", descending=True)

    return summary


def skill_progression_analysis(df: pl.DataFrame, aggregate_by_category: bool = False) -> Dict[str, pl.DataFrame]:
    """Analyze skill progression patterns

    Args:
        df: DataFrame with agent data including skills
        aggregate_by_category: If True, aggregate POIs by category

    Returns:
        Dictionary containing different skill analysis DataFrames
    """
    # This function would need actual skill data columns
    # For now, return empty structure
    return {
        "skill_trends": pl.DataFrame(),
        "poi_skill_correlation": pl.DataFrame(),
        "agent_skill_comparison": pl.DataFrame(),
    }


def create_poi_heatmap(memory_df: pl.DataFrame, pois_metadata: Dict[str, Dict]) -> go.Figure:
    """
    Create a heatmap of POI visits by time of day

    Args:
        memory_df: Temporal memory DataFrame
        pois_metadata: Dictionary mapping POI IDs to metadata including category

    Returns:
        Plotly figure
    """
    # Extract hour from time and add POI category
    df = memory_df.with_columns(
        [
            pl.col("time").str.slice(11, 13).cast(pl.Int64).alias("hour"),
        ]
    )

    # Add POI category
    df = df.with_columns(
        [
            pl.col("poi_id")
            .map_dict({poi_id: meta["category"] for poi_id, meta in pois_metadata.items()})
            .alias("category")
        ]
    )

    # Count visits by hour and category
    heatmap_data = (
        df.filter(pl.col("poi_id").is_not_null())
        .groupby(["hour", "category"])
        .count()
        .pivot(
            values="count",
            index="hour",
            columns="category",
        )
        .sort("hour")
    )

    # Convert to format needed for plotly heatmap
    matrix = heatmap_data.to_pandas().set_index("hour")

    # Create heatmap
    fig = px.imshow(
        matrix.T,
        labels=dict(x="Hour of Day", y="POI Category", color="Visit Count"),
        x=matrix.index,
        y=matrix.columns,
        color_continuous_scale="Viridis",
        title="POI Visits by Category and Time of Day",
    )

    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        width=800,
        height=600,
    )

    return fig


def plot_energy_over_time(daily_energy_df: pl.DataFrame) -> plt.Figure:
    """
    Plot energy levels over time

    Args:
        daily_energy_df: DataFrame from aggregate_daily_energy()

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    dates = daily_energy_df["date"].to_list()
    avg_energy = daily_energy_df["avg_energy"].to_list()
    min_energy = daily_energy_df["min_energy"].to_list()
    max_energy = daily_energy_df["max_energy"].to_list()

    ax.plot(dates, avg_energy, label="Average Energy", marker="o")
    ax.fill_between(dates, min_energy, max_energy, alpha=0.2, label="Min-Max Range")

    ax.set_xlabel("Date")
    ax.set_ylabel("Energy Level")
    ax.set_title("Agent Energy Levels Over Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig
