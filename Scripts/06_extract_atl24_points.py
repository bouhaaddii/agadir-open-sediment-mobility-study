from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")


def read_group(group: h5py.Group, key: str, n: int) -> np.ndarray:
    if key not in group:
        return np.full(n, np.nan, dtype="float64")
    values = np.asarray(group[key][...]).squeeze()
    if values.ndim != 1 or len(values) != n:
        return np.full(n, np.nan, dtype="float64")
    return values


def main() -> None:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    box = cfg["aoi_bbox_wgs84"]
    source = ROOT / "Data/Raw/ATL24"
    files = sorted([*source.rglob("*.h5"), *source.rglob("*.hdf5")])
    if not files:
        raise SystemExit("No ATL24 HDF5 granules found. Run Scripts/01_atl24_inventory_download.py --download first.")
    transformer = Transformer.from_crs("EPSG:4326", cfg["crs_projected"], always_xy=True)
    chunks: list[pd.DataFrame] = []
    count_in = 0
    for file_no, path in enumerate(files, 1):
        with h5py.File(path, "r") as h5:
            for beam in BEAMS:
                if beam not in h5:
                    continue
                g = h5[beam]
                if "class_ph" not in g or "lat_ph" not in g or "lon_ph" not in g or "ortho_h" not in g or "index_ph" not in g:
                    continue
                cls = np.asarray(g["class_ph"][...]).squeeze()
                if cls.ndim != 1:
                    continue
                n = len(cls)
                lat = read_group(g, "lat_ph", n).astype("float64")
                lon = read_group(g, "lon_ph", n).astype("float64")
                ortho = read_group(g, "ortho_h", n).astype("float64")
                conf = read_group(g, "confidence", n).astype("float64")
                surface = read_group(g, "surface_h", n).astype("float64")
                delta = read_group(g, "delta_time", n).astype("float64")
                photon_index = read_group(g, "index_ph", n).astype("float64")
                sigma_h = read_group(g, "sigma_thu", n).astype("float64")
                sigma_v = read_group(g, "sigma_tvu", n).astype("float64")
                keep = (
                    (cls == int(cfg["atl24_class_bathymetry"]))
                    & (conf >= float(cfg["atl24_confidence_min"]))
                    & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(ortho)
                    & np.isfinite(surface) & np.isfinite(delta) & np.isfinite(photon_index) & (photon_index >= 0)
                    & (lat >= box["south"]) & (lat <= box["north"])
                    & (lon >= box["west"]) & (lon <= box["east"])
                )
                idx = np.flatnonzero(keep)
                if not len(idx):
                    continue
                depth = surface[idx] - ortho[idx]
                valid_depth = np.isfinite(depth) & (depth > 0) & (depth <= 100)
                idx, depth = idx[valid_depth], depth[valid_depth]
                if not len(idx):
                    continue
                x, y = transformer.transform(lon[idx], lat[idx])
                # ATL24 delta_time is reported as seconds since the ATLAS product epoch (2018-01-01).
                # Keep the source timestamp interpretation explicit; scene matching allows a generous window.
                times = [
                    (datetime(2018, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(v))).isoformat().replace("+00:00", "Z")
                    for v in delta[idx]
                ]
                chunks.append(pd.DataFrame({
                    "granule_id": path.name,
                    "beam": beam,
                    "source_photon_index": photon_index[idx].astype("int64"),
                    "utc_time_source_epoch": times,
                    "longitude_deg": lon[idx], "latitude_deg": lat[idx],
                    "easting_m": x, "northing_m": y,
                    "ortho_height_egm08_m": ortho[idx],
                    "surface_height_egm08_m": surface[idx],
                    "instantaneous_depth_m": depth,
                    "confidence": conf[idx],
                    "sigma_thu_1sigma_m": sigma_h[idx],
                    "sigma_tvu_1sigma_m": sigma_v[idx],
                }))
                count_in += len(idx)
        print(f"Read {file_no}/{len(files)} ATL24 granules; retained points so far={count_in:,}")
    if not chunks:
        raise SystemExit("No ATL24 points passed the local AOI, class, confidence and depth filters. Inspect inventory/version and config.")
    out = ROOT / "Data/Processed/ATL24/atl24_bathy_points.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat(chunks, ignore_index=True)
    table.to_csv(out, index=False, compression="gzip")
    metadata = {
        "source_product": "NASA NSIDC ATL24 Version 003",
        "source_class": int(cfg["atl24_class_bathymetry"]),
        "confidence_min": float(cfg["atl24_confidence_min"]),
        "aoi_bbox_wgs84": box,
        "retained_points": int(len(table)),
        "height_reference": "EGM08 orthometric heights; instantaneous depth = surface_h - ortho_h",
        "photon_identifier": "source_photon_index is ATL24 index_ph inherited from the ATL03 photon index; use granule_id + beam + source_photon_index to identify a photon.",
        "warning": "ATL24 track points are sparse and time-varying. Do not interpret them as a continuous, contemporaneous bathymetric surface.",
    }
    (out.parent / "atl24_extraction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Retained ATL24 bathymetry photons: {len(table):,}")
    print(f"Points: {out}")
    print(f"Metadata: {out.parent / 'atl24_extraction_metadata.json'}")
    print("Depth is referenced to each measured sea surface; it is not a national-datum bathymetric elevation.")


if __name__ == "__main__":
    main()



