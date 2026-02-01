"""Plot rainfall (AWS IIT Mandi) and river discharge (Kamand_1).

Reads the two filtered CSVs in `data/filtered/`, filters only the specified stations,
parses the the `Data Acquisition Time` as datetime (day-first), and plots the
rainfall (mm) and discharge (m3/sec) on a shared time axis with twin y-axes.

Usage:
    uv run data_plot.py            # creates plot and saves to ./plots/aws_kamand.png and shows it
    uv run data_plot.py --no-show  # save without showing interactively

Notes:
    Showing the plot uses the PyQt5 backend (Qt5Agg). Install PyQt5 if you plan to show plots: `pip install PyQt5`.
    Prefer `uv run <script>` so the script runs using the workspace's configured environment (virtualenv/pyproject).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib as mpl
import pandas as pd


DATA_DIR = Path(__file__).parent / "data" / "filtered"
RAIN_FILE = DATA_DIR / "rainfall_tel_hr_himachal_pradesh_hp_2021_2025.csv"
RIVER_FILE = DATA_DIR / "river_discharge_tele_hr_himachal_pradesh_hp_1970_2025.csv"


def read_and_filter_rainfall(path: Path, station: str = "AWS IIT Mandi") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Data Acquisition Time" not in df.columns:
        raise KeyError("expected 'Data Acquisition Time' column in rainfall file")

    df["datetime"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce")
    df = df[df["Station"] == station].copy()
    df = df.dropna(subset=["datetime"])  # drop any bad-parsed rows
    df = df.sort_values("datetime")
    return df


def read_and_filter_river(path: Path, station: str = "Kamand_1") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Data Acquisition Time" not in df.columns:
        raise KeyError("expected 'Data Acquisition Time' column in river discharge file")

    df["datetime"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce")
    df = df[df["Station"] == station].copy()
    df = df.dropna(subset=["datetime"])  # drop any bad-parsed rows
    df = df.sort_values("datetime")
    return df


def remove_outliers(df: pd.DataFrame, column: str, method: str = "iqr", iqr_factor: float = 1.5, z_threshold: float = 3.0) -> pd.DataFrame:
    """Return a copy of `df` with outliers removed from `column`.

    Methods:
    - 'iqr' : remove values outside [Q1 - iqr_factor*IQR, Q3 + iqr_factor*IQR]
    - 'zscore' : remove values with abs(z) > z_threshold
    Non-numeric values are treated as NaN and preserved (not considered outliers).
    """
    s = pd.to_numeric(df[column], errors="coerce")
    if method == "iqr":
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        mask = ((s >= lower) & (s <= upper)) | s.isna()
    elif method == "zscore":
        mean = s.mean()
        std = s.std()
        if pd.isna(std) or std == 0:
            mask = s.notna()
        else:
            z = (s - mean) / std
            mask = (z.abs() <= z_threshold) | s.isna()
    else:
        raise ValueError(f"unknown outlier removal method: {method}")
    return df.loc[mask].copy()


def plot_kandhi_bajoura(
    rain_path: Path = RAIN_FILE,
    river_path: Path = RIVER_FILE,
    out_path: Path | None = None,
    show: bool = True,
    remove_outliers_flag: bool = True,
    outlier_method: str = "iqr",
    iqr_factor: float = 1.5,
    z_threshold: float = 3.0,
) -> Path:
    """Read data, filter required stations, remove outliers by default, and plot. Returns path to saved image.

    Outliers are removed by default from the value columns. Use --no-remove-outliers to disable.
    """
    rain = read_and_filter_rainfall(rain_path, station="AWS IIT Mandi")
    river = read_and_filter_river(river_path, station="Kamand_1")

    if rain.empty:
        raise ValueError("No rainfall records found for station 'AWS IIT Mandi'.")
    if river.empty:
        raise ValueError("No river discharge records found for station 'Kamand_1'.")

    # Expected value columns from the filtered CSVs
    rain_col = "Telemetry Hourly Rainfall (mm)"
    river_col = "Telemetry Hourly River Water Discharge (m3/sec)"
    if rain_col not in rain.columns:
        raise KeyError(f"expected '{rain_col}' in rainfall file")
    if river_col not in river.columns:
        raise KeyError(f"expected '{river_col}' in river file")

    # Ensure numeric values
    rain[rain_col] = pd.to_numeric(rain[rain_col], errors="coerce")
    river[river_col] = pd.to_numeric(river[river_col], errors="coerce")

    # Report counts and remove outliers if enabled
    rain_before = len(rain)
    river_before = len(river)
    if remove_outliers_flag:
        rain = remove_outliers(rain, rain_col, method=outlier_method, iqr_factor=iqr_factor, z_threshold=z_threshold)
        river = remove_outliers(river, river_col, method=outlier_method, iqr_factor=iqr_factor, z_threshold=z_threshold)
        print(f"Outlier removal applied: rainfall {rain_before} -> {len(rain)} rows; river {river_before} -> {len(river)} rows (method={outlier_method})")

    # Drop rows with missing datetimes or values before plotting
    rain_plot = rain.dropna(subset=["datetime", rain_col])
    river_plot = river.dropna(subset=["datetime", river_col])

    # If interactive display requested, ensure PyQt5 backend (Qt5Agg) is used
    if show:
        try:
            import PyQt5  # validate availability
        except Exception as e:
            raise RuntimeError("Showing plots requires PyQt5. Install via 'pip install PyQt5'.") from e
        mpl.use("Qt5Agg", force=True)
    # Import pyplot after backend is potentially set
    import matplotlib.pyplot as plt

    # Plot
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_title("Rainfall (AWS IIT Mandi) and River Discharge (Kamand_1)")
    ax1.set_xlabel("Datetime")

    ax1.plot(rain_plot["datetime"], rain_plot[rain_col], color="tab:blue", marker="o", linestyle="-", label="Rainfall (mm)")
    ax1.set_ylabel("Rainfall (mm)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(river_plot["datetime"], river_plot[river_col], color="tab:orange", linestyle="-", alpha=0.9, label="Discharge (m3/s)")
    ax2.set_ylabel("River Discharge (m3/sec)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    # Beautify x-axis
    fig.autofmt_xdate(rotation=30)
    ax1.grid(True, which="both", axis="x", linestyle="--", linewidth=0.4)

    # Save plot
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    out_path = out_path or out_dir / "aws_kamand.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)

    # Show plot (default behavior: show along with saving unless disabled)
    if show:
        plt.show()
    plt.close(fig)
    return Path(out_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot AWS IIT Mandi rainfall and Kamand_1 discharge")
    p.add_argument("--rain", type=Path, default=RAIN_FILE, help="rainfall CSV path")
    p.add_argument("--river", type=Path, default=RIVER_FILE, help="river discharge CSV path")
    p.add_argument("--out", type=Path, default=None, help="output image file path")
    p.add_argument("--no-remove-outliers", action="store_true", help="do not remove outliers from the value columns before plotting (default: remove)")
    p.add_argument("--outlier-method", choices=["iqr", "zscore"], default="iqr", help="method to remove outliers")
    p.add_argument("--outlier-iqr-factor", type=float, default=1.5, help="IQR multiplier for outlier detection (iqr method)")
    p.add_argument("--outlier-z-threshold", type=float, default=3.0, help="z-score threshold for outlier detection (zscore method)")
    p.add_argument("--no-show", action="store_true", help="do not show the plot interactively after saving")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    show = not getattr(args, "no_show", False)
    out = plot_kandhi_bajoura(
        rain_path=args.rain,
        river_path=args.river,
        out_path=args.out,
        show=show,
        remove_outliers_flag=not getattr(args, "no_remove_outliers", False),
        outlier_method=args.outlier_method,
        iqr_factor=args.outlier_iqr_factor,
        z_threshold=args.outlier_z_threshold,
    )
    print(f"Saved plot to: {out}")
