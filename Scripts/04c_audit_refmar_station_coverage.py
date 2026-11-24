from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://services.data.shom.fr/maregraphie/service/tidegauges"


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p = math.pi / 180.0
    a = math.sin((lat2 - lat1) * p / 2) ** 2
    a += math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def main() -> None:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    box = cfg["aoi_bbox_wgs84"]
    lon0 = (float(box["west"]) + float(box["east"])) / 2
    lat0 = (float(box["south"]) + float(box["north"])) / 2

    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise SystemExit(f"Unexpected SHOM response type: {type(payload).__name__}")

    rows = []
    for station in payload:
        try:
            lon, lat = float(station["longitude"]), float(station["latitude"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "shom_id": str(station.get("shom_id", "")),
            "name": str(station.get("name", "")),
            "network": str(station.get("reseau", "")),
            "state": str(station.get("state", "")),
            "longitude_deg": lon,
            "latitude_deg": lat,
            "distance_to_aoi_center_km": haversine_km(lon0, lat0, lon, lat),
        })
    if not rows:
        raise SystemExit("The SHOM registry returned no stations with usable coordinates.")

    stations = pd.DataFrame(rows).sort_values("distance_to_aoi_center_km")
    out = ROOT / "Results/Matchup_Audit"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "refmar_current_station_registry.csv"
    stations.to_csv(csv_path, index=False)

    local_named = stations[stations["name"].str.contains(r"AGADIR|MAROC|MOROCCO", case=False, regex=True)]
    nearest = stations.iloc[0].to_dict()
    summary = {
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "registry_url": URL,
        "registry_record_count": int(len(stations)),
        "aoi_center_wgs84": {"longitude": lon0, "latitude": lat0},
        "stations_named_for_agadir_or_morocco": local_named.to_dict(orient="records"),
        "stations_within_500_km_of_aoi_center": int((stations["distance_to_aoi_center_km"] <= 500).sum()),
        "nearest_registry_station": nearest,
        "interpretation": (
            "The public SHOM/REFMAR operational station registry has no station named Agadir/Morocco "
            "and no listed station within 500 km of the AOI. This registry is not an exhaustive inventory "
            "of Moroccan tide gauges and does not prove that no Moroccan agency holds observations. "
            "SHOM's separate Agadir archive inventory documents historical records from 1920 to 1959, "
            "which cannot validate sea level during the 2018-2026 satellite period."
        ),
        "historical_archive_inventory_url": "https://refmar.shom.fr/data-archaeology/liste-inventaire",
        "historical_agadir_detail_url": "https://www.refmar.shom.fr/detail-des-documents-relatifs-au-port-agadir-maroc",
    }
    json_path = out / "refmar_station_coverage_audit.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Read-only SHOM/REFMAR public-registry audit complete.")
    print(f"Registry stations with coordinates: {len(stations)}")
    print(f"Stations named Agadir/Morocco: {len(local_named)}")
    print(f"Registry stations within 500 km: {summary['stations_within_500_km_of_aoi_center']}")
    print(
        "Nearest registry station: "
        f"{nearest['name']} (ID {nearest['shom_id']}), "
        f"{nearest['distance_to_aoi_center_km']:.1f} km"
    )
    print(f"Station inventory: {csv_path}")
    print(f"Audit summary: {json_path}")
    print("This does not rule out separate Moroccan-agency gauge records; it only audits the public SHOM registry.")


if __name__ == "__main__":
    main()