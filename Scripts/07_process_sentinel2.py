from __future__ import annotations

import csv
import json
import re
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("blue", "green", "red", "nir", "scl")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_to_template(path: Path, template: rasterio.io.DatasetReader, resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as src:
        out = np.full((template.height, template.width), src.nodata if src.nodata is not None else 0, dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=template.transform, dst_crs=template.crs,
            src_nodata=src.nodata, dst_nodata=src.nodata if src.nodata is not None else 0,
            resampling=resampling,
        )
        return out


def reflectance_scale(src: rasterio.io.DatasetReader, arr: np.ndarray) -> float:
    scale = float(src.scales[0]) if src.scales else 1.0
    if 0 < scale < 0.01:
        return scale
    tags = src.tags(1)
    for key in ("scale", "reflectance_scale", "scale_factor"):
        try:
            value = float(tags.get(key, ""))
            if 0 < value < 0.01:
                return value
        except (TypeError, ValueError):
            pass
    return 1e-4 if np.issubdtype(arr.dtype, np.integer) and np.nanmax(arr) > 2 else 1.0


def save_float(path: Path, values: np.ndarray, template: rasterio.io.DatasetReader, description: str) -> None:
    profile = template.profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate", tiled=True)
    data = np.where(np.isfinite(values), values, -9999.0).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process selected Sentinel-2 reflectance and SCL assets.")
    parser.add_argument("--selection-file", default=None, help="Optional selection CSV relative to project root or an absolute path.")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    table = Path(args.selection_file) if args.selection_file else ROOT / "Data/Manifests/sentinel2_selection.csv"
    if not table.is_absolute():
        table = ROOT / table
    if not table.exists():
        raise SystemExit("Run Scripts/04_catalog_sentinel2.py first.")
    with table.open(newline="", encoding="utf-8-sig") as stream:
        selected = [r for r in csv.DictReader(stream) if r.get("selected", "").strip().upper() == "YES"]
    if not selected:
        raise SystemExit("No Sentinel-2 items are marked YES. Edit the selection CSV and rerun.")
    raw = ROOT / "Data/Raw/Sentinel2"
    outdir = ROOT / "Data/Processed/Sentinel2"
    outdir.mkdir(parents=True, exist_ok=True)
    processed = 0
    missing_scenes = []
    for i, row in enumerate(selected, 1):
        date = row["acquisition_date"].replace("-", "")
        stem = safe_name(row["item_id"])
        paths = {a: raw / f"{date}_{stem}_{a}.tif" for a in ASSETS}
        missing = [a for a, p in paths.items() if not p.exists()]
        if missing:
            print(f"{i}/{len(selected)} {row['acquisition_date']}: missing {', '.join(missing)}; run Script 05 first.")
            missing_scenes.append((row["item_id"], missing))
            continue
        with rasterio.open(paths["green"]) as template:
            values = {}
            for name in ("green", "blue", "red", "nir"):
                with rasterio.open(paths[name]) as src:
                    raw_values = read_to_template(paths[name], template, Resampling.bilinear)
                    values[name] = raw_values.astype("float32") * reflectance_scale(src, raw_values)
            green, blue, red, nir = values["green"], values["blue"], values["red"], values["nir"]
            scl = read_to_template(paths["scl"], template, Resampling.nearest).astype("int16")
            finite = np.isfinite(blue) & np.isfinite(green) & np.isfinite(red) & np.isfinite(nir)
            # Apply the raster scale when it is supplied; otherwise infer the standard integer-DN scale.
            water = (scl == 6) & finite & (green + nir != 0) & (red + green != 0)
            ndwi = np.full(green.shape, np.nan, dtype="float32")
            ndti = np.full(green.shape, np.nan, dtype="float32")
            ndwi[water] = (green[water] - nir[water]) / (green[water] + nir[water])
            ndti[water] = (red[water] - green[water]) / (red[water] + green[water])
            prefix = outdir / f"{date}_{stem}"
            save_float(Path(str(prefix) + "_NDWI.tif"), ndwi, template, "NDWI; SCL water pixels only")
            save_float(Path(str(prefix) + "_NDTI.tif"), ndti, template, "NDTI; SCL water pixels only; relative proxy")
            mask_profile = template.profile.copy()
            mask_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate", tiled=True)
            with rasterio.open(Path(str(prefix) + "_watermask.tif"), "w", **mask_profile) as dst:
                dst.write(water.astype("uint8"), 1)
                dst.set_band_description(1, "SCL class 6 water with finite reflectance")
            print(f"{i}/{len(selected)} {row['acquisition_date']}: clear-water pixels={int(water.sum()):,}")
            processed += 1
    if missing_scenes:
        raise SystemExit(f"Processing incomplete: {processed}/{len(selected)} scenes processed; run Script 05 to fetch missing assets, then rerun this script.")
    print(f"Processed index and mask rasters: {outdir}")
    print("NDTI is a relative optical proxy; it is not a suspended-sediment concentration retrieval.")


if __name__ == "__main__":
    main()
