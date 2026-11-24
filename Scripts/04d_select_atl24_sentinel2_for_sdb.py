from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "Data/Manifests/sentinel2_selection.csv"
MATCHUPS = ROOT / "Results/Matchup_Audit/sentinel2_atl24_temporal_matchup_audit.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an isolated Sentinel-2 selection for ATL24 SDB matchups.")
    parser.add_argument("--window-hours", type=int, default=6, choices=(1, 3, 6, 12, 24))
    parser.add_argument("--output", default=None, help="Output CSV path relative to the project root or an absolute path.")
    args = parser.parse_args()
    if not CATALOG.exists() or not MATCHUPS.exists():
        raise SystemExit("Run Scripts/04_catalog_sentinel2.py and Scripts/04b_audit_atl24_sentinel2_matchups.py first.")

    catalog = pd.read_csv(CATALOG, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    audit = pd.read_csv(MATCHUPS, encoding="utf-8-sig")
    required = {"item_id", "acquisition_date", "acquisition_utc", "assets_json", "selected"}
    if missing := required.difference(catalog.columns):
        raise SystemExit(f"Sentinel-2 catalogue lacks columns: {sorted(missing)}")
    candidate_col = f"temporal_candidate_pm_{args.window_hours}h"
    gap_col = "nearest_photon_gap_min"
    if candidate_col not in audit or gap_col not in audit:
        raise SystemExit(f"Matchup audit lacks {candidate_col} or {gap_col}; rerun Script 04b.")

    candidates = audit.loc[audit[candidate_col].astype(str).str.lower().isin(("true", "1"))].copy()
    if candidates.empty:
        raise SystemExit(f"No catalogue items have ATL24 photons within ±{args.window_hours} h.")
    candidates[gap_col] = pd.to_numeric(candidates[gap_col], errors="coerce")
    # One product per acquisition date avoids duplicate same-day scene weighting.
    # Prefer the closest ATL24 photon; item_id provides deterministic tie-breaking.
    candidates = candidates.sort_values([gap_col, "acquisition_date", "item_id"], kind="stable")
    candidates = candidates.drop_duplicates("acquisition_date", keep="first")
    selected_ids = set(candidates["item_id"].astype(str))

    out = catalog.copy()
    out["selected"] = out["item_id"].astype(str).map(lambda item: "YES" if item in selected_ids else "NO")
    extra = candidates.set_index("item_id")[[gap_col, "nearest_atl24_granule"]].to_dict("index")
    out["atl24_nearest_photon_gap_min"] = out["item_id"].map(lambda item: extra.get(str(item), {}).get(gap_col, ""))
    out["atl24_nearest_granule"] = out["item_id"].map(lambda item: extra.get(str(item), {}).get("nearest_atl24_granule", ""))
    out_path = Path(args.output) if args.output else ROOT / f"Data/Manifests/sentinel2_atl24_pm{args.window_hours}h_selection.csv"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    selected = out[out["selected"] == "YES"]
    print("ATL24-matched Sentinel-2 selection prepared; the original catalogue was not modified.")
    print(f"Window: ±{args.window_hours} h; selected scenes: {len(selected)}; distinct dates: {selected['acquisition_date'].nunique()}")
    print(f"Selection file: {out_path}")
    print("This is a temporal screening list only. Inspect per-pixel SCL, clear-water coverage, spatial overlap, and tide-model support before fitting SDB.")


if __name__ == "__main__":
    main()
