from __future__ import annotations

import csv
import json
import re
import argparse
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ["blue", "green", "red", "nir", "swir16", "swir22", "scl"]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AOI-clipped Sentinel-2 assets for selected catalogue rows.")
    parser.add_argument("--selection-file", default=None, help="Optional selection CSV relative to project root or an absolute path.")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    table = Path(args.selection_file) if args.selection_file else ROOT / "Data/Manifests/sentinel2_selection.csv"
    if not table.is_absolute():
        table = ROOT / table
    if not table.exists():
        raise FileNotFoundError("Run Scripts/04_catalog_sentinel2.py first.")
    with table.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("selected", "").strip().upper() == "YES"]
    if not rows:
        print(f"No Sentinel-2 row is marked YES in {table}; nothing downloaded.")
        return
    box = cfg["aoi_bbox_wgs84"]
    outdir = ROOT / "Data/Raw/Sentinel2"
    outdir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for i, row in enumerate(rows, start=1):
        assets = json.loads(row["assets_json"])
        stamp = row["acquisition_date"].replace("-", "")
        stem = safe_name(row["item_id"])
        print(f"{i}/{len(rows)} {row['acquisition_date']} {row['item_id']}")
        for asset_name in ASSETS:
            href = assets.get(asset_name)
            if not href:
                print(f"  Missing asset {asset_name}; skipped.")
                continue
            target = outdir / f"{stamp}_{stem}_{asset_name}.tif"
            if target.exists() and target.stat().st_size > 1000:
                print(f"  Present; skipped {asset_name}.")
                continue
            with rasterio.open(href) as src:
                projected = transform_bounds("EPSG:4326", src.crs, box["west"], box["south"], box["east"], box["north"], densify_pts=21)
                window = from_bounds(*projected, transform=src.transform).round_offsets().round_lengths()
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                if window.width <= 0 or window.height <= 0:
                    print(f"  AOI does not intersect asset {asset_name}; skipped.")
                    continue
                array = src.read(window=window)
                profile = src.profile.copy()
                profile.update(height=int(window.height), width=int(window.width), transform=src.window_transform(window), compress="deflate")
                with rasterio.open(target, "w", **profile) as dst:
                    dst.write(array)
                    dst.update_tags(source_href=href, stac_item=row["item_id"], acquisition_utc=row["acquisition_utc"], asset=asset_name)
            n_written += 1
            print(f"  Wrote {asset_name}: {target.stat().st_size / 1e6:.2f} MB")
    print(f"Finished: {n_written} band files written under {outdir}")
    print("Original COGs were read by HTTP range requests; local files are AOI-clipped source values.")


if __name__ == "__main__":
    main()
