from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m"
VARIABLE = "zos"
FILENAME = "IBI_ssh_Anza_hourly.nc"


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description="Download hourly IBI sea-surface height for relative S2/ATL24 tide matching.")
    parser.add_argument("--dry-run", action="store_true", help="Validate subset and transfer size without downloading data.")
    parser.add_argument("--download", action="store_true", help="Submit the subset request; combine with --dry-run for preview only.")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    outdir = ROOT / "Data/Raw/Forcing"
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / FILENAME
    if output.exists() and not args.dry_run:
        print(f"Existing file preserved: {output} ({output.stat().st_size / 1e6:.2f} MB)")
        print("Remove it manually only if you intentionally want a fresh copy; this script will not overwrite it.")
        return

    cli_name = "copernicusmarine.exe" if os.name == "nt" else "copernicusmarine"
    cli_path = Path(sys.executable).resolve().parent / cli_name
    if not cli_path.is_file():
        raise SystemExit(f"Copernicus Marine CLI not found: {cli_path}. Install it in this environment first.")
    box = cfg["aoi_bbox_wgs84"]
    start = cfg["date_start"] + "T00:00:00"
    end = cfg["date_end"] + "T23:00:00"
    command = [
        str(cli_path), "subset", "--dataset-id", DATASET,
        "--variable", VARIABLE,
        "--minimum-longitude", str(box["west"]), "--maximum-longitude", str(box["east"]),
        "--minimum-latitude", str(box["south"]), "--maximum-latitude", str(box["north"]),
        "--start-datetime", start, "--end-datetime", end,
        "--output-directory", str(outdir), "--output-filename", FILENAME,
    ]
    if args.dry_run:
        command.append("--dry-run")
    print("Dataset:", DATASET)
    print("Variable:", VARIABLE, "(hourly sea-surface height above geoid)")
    print("Time range (UTC):", start, "to", end)
    print("Approximate project box:", box)
    print("Output:", output)
    print("Use zos differences only for relative sea-level adjustment; the regional model is not a local gauge or datum tie.")
    if not args.download:
        print("Preview:", subprocess.list2cmdline(command))
        print("No data downloaded. Inspect this request, then run with --dry-run --download and finally --download.")
        return
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit("Copernicus Marine request failed; inspect the message above and rerun after resolving it.")
    if args.dry_run:
        print("Dry run complete; no file was downloaded.")
    else:
        print(f"Downloaded hourly SSH subset: {output}")
        print("Next: run Scripts/15_audit_ibi_ssh_matchups.py to check wet-cell distance and matchup support.")


if __name__ == "__main__":
    main()
