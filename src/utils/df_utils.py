from typing import Dict, List, Optional
import polars as pl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def append_records(df: pl.DataFrame, records: List[Dict]) -> pl.DataFrame:
    """Append multiple records to a DataFrame"""
    new_df = pl.DataFrame(records)
    return pl.concat([df, new_df])


def aggregate_daily_energy(
    memory_df: pl.DataFrame, agent_id: Optional[str] = None
) -> pl.DataFrame:
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
    filtered_df = filtered_df.with_columns(
        pl.col("time").str.slice(0, 10).alias("date")
    )

    # Group by date and get energy stats
    result = filtered_df.groupby("date").agg(
        pl.min("current_energy").alias("min_energy"),
        pl.mean("current_energy").alias("avg_energy"),
        pl.max("current_energy").alias("max_energy"),
        pl.count("current_energy").alias("actions"),
    )

    return result.sort("date")


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


def create_poi_heatmap(
    memory_df: pl.DataFrame, pois_metadata: Dict[str, Dict]
) -> go.Figure:
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
            .map_dict(
                {poi_id: meta["category"] for poi_id, meta in pois_metadata.items()}
            )
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
