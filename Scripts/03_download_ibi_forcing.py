from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def make_command(dataset: str, variables: list[str], box: dict, start: str, end: str, outdir: Path, filename: str, depths: tuple[float, float] | None = None, dry_run: bool = False) -> list[str]:
    cli_name = "copernicusmarine.exe" if os.name == "nt" else "copernicusmarine"
    cli_path = Path(sys.executable).resolve().parent / cli_name
    if not cli_path.is_file():
        raise SystemExit(
            f"Copernicus Marine command-line tool was not found at {cli_path}. "
            "Install it in this virtual environment with: "
            f'"{sys.executable}" -m pip install copernicusmarine'
        )
    command = [str(cli_path), "subset", "--dataset-id", dataset]
    for variable in variables:
        command.extend(["--variable", variable])
    command.extend([
        "--minimum-longitude", str(box["west"]), "--maximum-longitude", str(box["east"]),
        "--minimum-latitude", str(box["south"]), "--maximum-latitude", str(box["north"]),
        "--start-datetime", start + "T00:00:00", "--end-datetime", end + "T23:00:00",
        "--output-directory", str(outdir), "--output-filename", filename, "--overwrite",
    ])
    if depths:
        command.extend(["--minimum-depth", str(depths[0]), "--maximum-depth", str(depths[1])])
    if dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset regional Copernicus Marine IBI forcing. Authenticate first with copernicusmarine login.")
    parser.add_argument("--dry-run", action="store_true", help="Ask Copernicus Marine to validate subsets without downloading.")
    parser.add_argument("--download", action="store_true", help="Run the subsets. Without this flag only a command preview is printed.")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    outdir = ROOT / "Data/Raw/Forcing"
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (cfg["ibi_wave_dataset"], cfg["ibi_wave_variables"], "IBI_waves_Anza_hourly.nc", None),
        (cfg["ibi_current_dataset"], cfg["ibi_current_variables"], "IBI_currents_Anza_daily_0_30m.nc", (cfg["current_min_depth_m"], cfg["current_max_depth_m"])),
    ]
    for dataset, variables, filename, depths in jobs:
        command = make_command(dataset, variables, cfg["aoi_bbox_wgs84"], cfg["date_start"], cfg["date_end"], outdir, filename, depths, args.dry_run)
        print("Dataset:", dataset)
        print("Variables:", ", ".join(variables))
        print("Window:", cfg["date_start"], "to", cfg["date_end"])
        if depths:
            print("Depth request:", depths[0], "to", depths[1], "m")
        print("Output:", outdir / filename)
        if not args.download:
            print("Preview:", subprocess.list2cmdline(command))
            continue
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(f"Copernicus Marine subset failed for {dataset}; correct the account or dataset message above and rerun.")
    if not args.download:
        print("No data downloaded. After reviewing the subset, run with --dry-run --download, then run with --download.")
    else:
        print("IBI subsets completed. Regional resolution does not resolve the surf zone.")


if __name__ == "__main__":
    main()
