from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import earthaccess

ROOT = Path(__file__).resolve().parents[1]


def load_downloader():
    path = ROOT / "Scripts/01_atl24_inventory_download.py"
    spec = importlib.util.spec_from_file_location("atl24_downloader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Show live ATL24 download progress without changing files.")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds (default: 5).")
    args = parser.parse_args()
    if args.interval < 1:
        raise SystemExit("Use a refresh interval of at least one second.")

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    box = cfg["aoi_bbox_wgs84"]
    granules = earthaccess.search_data(
        short_name="ATL24",
        version=cfg["atl24_version"],
        bounding_box=(box["west"], box["south"], box["east"], box["north"]),
        temporal=(cfg["date_start"], cfg["date_end"]),
    )
    downloader = load_downloader()
    target = ROOT / "Data/Raw/ATL24"
    expected = []
    for granule in granules:
        links = granule.data_links()
        if not links:
            continue
        size_bytes = downloader.expected_size_bytes(granule)
        if size_bytes:
            expected.append((Path(urlparse(links[0]).path).name, size_bytes))
    total_bytes = sum(size for _, size in expected)
    if not total_bytes:
        raise SystemExit("No expected granule sizes were returned by the ATL24 catalogue.")

    print(f"Suivi en lecture seule de {len(expected)} granules; actualisation toutes les {args.interval:g} s.")
    print("Arrêter ce moniteur avec Ctrl+C n’interrompt pas le téléchargement dans l’autre fenêtre.")
    try:
        while True:
            received = 0
            present = 0
            at_size = 0
            for filename, size in expected:
                path = target / filename
                partial_path = path.with_suffix(path.suffix + ".part")
                source = path if path.is_file() else partial_path
                if not source.is_file():
                    continue
                present += 1
                local_size = source.stat().st_size
                received += min(local_size, size)
                tolerance = max(1024**2, int(size * 0.01))
                if local_size >= size - tolerance:
                    at_size += 1
            percent = min(100.0, received * 100.0 / total_bytes)
            bar_width = 28
            filled = round(bar_width * percent / 100.0)
            bar = "#" * filled + "." * (bar_width - filled)
            status = (
                f"[{bar}] {percent:5.1f}% | {received / 1024**3:5.2f}/"
                f"{total_bytes / 1024**3:.2f} Gio | fichiers à taille attendue: "
                f"{at_size}/{len(expected)} | présents: {present}/{len(expected)}"
            )
            sys.stdout.write("\r" + status.ljust(150))
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        sys.stdout.write("\nMoniteur arrêté; le téléchargement dans l’autre fenêtre continue.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
