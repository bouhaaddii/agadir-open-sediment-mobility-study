from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["numpy", "pandas", "scipy", "xarray", "h5py", "rasterio", "pyproj", "pystac_client", "requests", "copernicusmarine", "earthaccess", "sklearn", "matplotlib", "docx"]


def main() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    usage = shutil.disk_usage(ROOT)
    print("Open-source Anza–Agadir project preflight")
    print(f"Project: {ROOT}")
    print(f"Configured AOI WGS84: {config['aoi_bbox_wgs84']}")
    print(f"Available disk space: {usage.free / 1e9:.1f} GB")
    if missing:
        print("Missing Python packages: " + ", ".join(missing))
        print('Install them with: & ".\\.venv\\Scripts\\python.exe" -m pip install -r ".\\requirements.txt"')
        raise SystemExit(2)
    print(f"All {len(REQUIRED)} required Python packages are available.")
    for folder in ["Data/Raw/ATL24", "Data/Raw/Bathymetry", "Data/Raw/Forcing", "Data/Raw/Sentinel2", "Data/Processed/ATL24", "Data/Processed/Sentinel2", "Data/Manifests", "Results/QC", "Results/Bathymetry", "Results/Sediment_Mobility"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    print("Project folders are ready.")


if __name__ == "__main__":
    main()
