from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the standard GOT5.5 base and GOT5.6 extension required by pyTMD's GOT5.6 model.")
    parser.add_argument("--download", action="store_true", help="Download required GOT5.5/GOT5.6 constituents. Without this flag, only show the plan.")
    args = parser.parse_args()
    target = ROOT / "Data/Raw/Tide_Models"
    print("Tide model: GOT5.6 standard ocean constituents (GOT5.5 base + GOT5.6 extension)")
    print(f"Source: NASA GSFC via pyTMD; directory: {target}")
    print("This is a harmonic tide model; it does not include storm surge or wave setup/runup.")
    if not args.download:
        print("Required files: GOT5.5 base constituents and GOT5.6 extension; no coastal extrapolation is requested.")
        print("No model data downloaded. After reviewing, run this script with --download.")
        return
    try:
        import pyTMD
        from pyTMD.datasets import fetch_gsfc_got
    except ImportError as exc:
        raise SystemExit(
            "pyTMD is missing. Install the project dependencies first with: "
            ' .\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements.txt'
        ) from exc

    target.mkdir(parents=True, exist_ok=True)
    print(f"pyTMD version: {getattr(pyTMD, '__version__', 'unknown')}")
    # pyTMD's GOT5.6 database combines constituents stored under GOT5.5 and
    # GOT5.6. Read its declared paths first, so a complete local installation
    # is reused instead of attempting to unpack the remote archives again.
    database_path = Path(pyTMD.__file__).resolve().parent / "data" / "database.json"
    database = json.loads(database_path.read_text(encoding="utf-8"))
    relative_required = database["GOT5.6"]["z"]["model_file"]
    missing_relative = [rel for rel in relative_required if not (target / rel).is_file()]
    missing_sets = {Path(rel).parts[0] for rel in missing_relative}

    if "GOT5.5" in missing_sets:
        print("Some GOT5.5 base constituents are missing; downloading the base archive...")
        fetch_gsfc_got(model="GOT5.5", directory=target, format="netcdf", compressed=False, timeout=1800)
    else:
        print("GOT5.5 base constituents are already present; skipping that download.")

    if "GOT5.6" in missing_sets:
        print("Some GOT5.6 extension constituents are missing; downloading the extension archive...")
        fetch_gsfc_got(model="GOT5.6", directory=target, format="netcdf", compressed=False, timeout=1800)
    else:
        print("GOT5.6 extension constituents are already present; skipping that download.")

    # Resolve through pyTMD as a final integrity check.
    try:
        model = pyTMD.io.model(target).from_database("GOT5.6")
        required = [Path(path) for path in model.z.model_file]
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(
            "The GOT5.6 constituent set is incomplete or could not be resolved. "
            f"Rerun this script with --download. Detail: {exc}"
        ) from exc
    missing = [path for path in required if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise SystemExit(f"GOT5.6 is missing {len(missing)} required ocean constituents:\n{missing_text}")
    files = sorted(p for p in target.rglob("*.nc") if p.is_file())
    if not files:
        raise SystemExit("The downloader returned, but no GOT NetCDF constituent files were found.")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "GOT5.6",
        "source": "NASA GSFC, downloaded using pyTMD",
        "spatial_extrapolation_requested": False,
        "constituent_sets_downloaded": ["GOT5.5 base", "GOT5.6 extension"],
        "required_ocean_constituents": len(required),
        "required_ocean_constituents_present": len(required) - len(missing),
        "files": [],
    }
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        manifest["files"].append({
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    manifest_path = target / "GOT5.6_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"NetCDF files inventoried: {len(files)}")
    print(f"Required GOT5.6 ocean constituents present: {len(required)}/{len(required)}")
    print(f"Manifest: {manifest_path}")
    print("Next: run Scripts/08_audit_got56_against_atl24_surface.py to check wet-grid support and compare pass-level anomalies.")


if __name__ == "__main__":
    main()
