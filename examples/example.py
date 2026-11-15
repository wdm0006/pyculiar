import datetime
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from pyculiar import detect_ts

matplotlib.style.use("ggplot")

__author__ = "willmcginnis"

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "tests" / "raw_data.csv"


def main(output_dir=None, show=False):
    """Run the minute-granularity fixture; write raw.csv, results.csv and example.png to output_dir."""
    out = Path(output_dir) if output_dir is not None else HERE
    out.mkdir(parents=True, exist_ok=True)

    twitter_example_data = pd.read_csv(DATA, usecols=["timestamp", "count"])
    twitter_example_data["timestamp"] = twitter_example_data["timestamp"].map(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp()
    )
    results = detect_ts(
        twitter_example_data, max_anoms=0.05, alpha=0.001, direction="both", granularity="min", verbose=True
    )

    twitter_example_data["timestamp"] = pd.to_datetime(twitter_example_data["timestamp"], unit="s")
    twitter_example_data.to_csv(out / "raw.csv", index=False)
    results["anoms"].to_csv(out / "results.csv", index=False)

    f, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(twitter_example_data["timestamp"], twitter_example_data["count"], "b")
    ax[0].plot(results["anoms"].index, results["anoms"]["anoms"], "ro")
    ax[0].set_title("Detected Anomalies")
    ax[1].set_xlabel("Time Stamp")
    ax[0].set_ylabel("Count")
    ax[1].plot(results["anoms"].index, results["anoms"]["anoms"], "b")
    ax[1].set_ylabel("Anomaly Magnitude")
    plot_path = out / "example.png"
    f.savefig(plot_path)
    if show:
        plt.show()
    plt.close(f)
    return results, plot_path


if __name__ == "__main__":
    headless = sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if headless:
        matplotlib.use("Agg")
    main(output_dir=sys.argv[1] if len(sys.argv) > 1 else None, show=not headless)
