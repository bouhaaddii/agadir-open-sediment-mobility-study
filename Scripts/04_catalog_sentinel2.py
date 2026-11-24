from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

from pystac_client import Client


ROOT = Path(__file__).resolve().parents[1]
STAC_URL = "https://earth-search.aws.element84.com/v1"


def main() -> None:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    box = cfg["aoi_bbox_wgs84"]
    client = Client.open(STAC_URL)
    out = ROOT / "Data/Manifests/sentinel2_selection.csv"
    previous_selection: dict[str, str] = {}
    if out.is_file():
        with out.open(newline="", encoding="utf-8-sig") as stream:
            for old_row in csv.DictReader(stream):
                item_id = old_row.get("item_id", "")
                selected = old_row.get("selected", "").strip().upper()
                if item_id and selected in {"YES", "NO"}:
                    previous_selection[item_id] = selected
    rows = []
    start_year = int(cfg["date_start"][:4]); end_year = int(cfg["date_end"][:4])
    for year in range(start_year, end_year + 1):
        begin = max(cfg["date_start"], f"{year}-01-01")
        finish = min(cfg["date_end"], f"{year}-12-31")
        for attempt in range(4):
            try:
                search = client.search(
                    collections=[cfg["sentinel_collection"]],
                    bbox=[box["west"], box["south"], box["east"], box["north"]],
                    datetime=f"{begin}/{finish}",
                    max_items=None,
                )
                items = list(search.items())
                break
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"STAC query failed for {year}; retry after checking the network. Last error: {exc}") from exc
                wait = 2 ** (attempt + 1)
                print(f"STAC query for {year} failed; retrying in {wait}s ({type(exc).__name__}).")
                time.sleep(wait)
        print(f"{year}: {len(items)} catalogue items")
        for item in items:
            props = item.properties
            date = datetime.fromisoformat(props["datetime"].replace("Z", "+00:00"))
            assets = {key: asset.href for key, asset in item.assets.items()}
            rows.append({
                "item_id": item.id,
                "acquisition_utc": date.isoformat(),
                "acquisition_date": date.date().isoformat(),
                "platform": props.get("platform", ""),
                "tile_cloud_percent": props.get("eo:cloud_cover", ""),
                "selected": previous_selection.get(item.id, "NO"),
                "assets_json": json.dumps(assets, separators=(",", ":")),
            })
    dedup = {row["item_id"]: row for row in rows}
    rows = list(dedup.values())
    rows.sort(key=lambda row: (row["acquisition_utc"], row["item_id"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["item_id", "acquisition_utc", "acquisition_date", "platform", "tile_cloud_percent", "selected", "assets_json"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Catalogued Sentinel-2 L2A items: {len(rows)}")
    print(f"Selection table: {out}")
    print("No tile-cloud cutoff was applied; tile cloud percentage is retained as metadata.")
    print("Local SCL, pixel coverage, and scene QA are required before using a scene.")
    print("Mark the chosen rows selected=YES before running the download step.")


if __name__ == "__main__":
    main()
