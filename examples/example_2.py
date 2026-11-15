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
DATA = HERE.parent / "tests" / "midnight_test_data.csv"


def main(output_dir=None, show=False):
    """Run the hourly-granularity fixture; write example_2.png to output_dir."""
    out = Path(output_dir) if output_dir is not None else HERE
    out.mkdir(parents=True, exist_ok=True)

    example_data = pd.read_csv(DATA, usecols=["date", "value"], parse_dates=["date"])

    results = detect_ts(example_data, max_anoms=0.05, alpha=0.001, granularity="hr", direction="both")

    f, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(example_data["date"], example_data["value"], "b")
    ax[0].plot(results["anoms"].index, results["anoms"]["anoms"], "ro")
    ax[0].set_title("Detected Anomalies")
    ax[1].set_xlabel("Time Stamp")
    ax[0].set_ylabel("Value")
    ax[1].plot(results["anoms"].index, results["anoms"]["anoms"], "b")
    ax[1].set_ylabel("Anomaly Magnitude")
    plot_path = out / "example_2.png"
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
