from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://ows.emodnet-bathymetry.eu/wcs"


def main() -> None:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    box = cfg["aoi_bbox_wgs84"]
    out = ROOT / "Data/Raw/Bathymetry/emodnet_mean_2022_anza.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 10_000:
        print(f"Already present; skipped: {out} ({out.stat().st_size / 1e6:.2f} MB)")
        return
    params = [
        ("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
        ("COVERAGEID", cfg["emodnet_coverage"]),
        ("SUBSET", f"Long({box['west']},{box['east']})"),
        ("SUBSET", f"Lat({box['south']},{box['north']})"),
        ("FORMAT", "image/tiff"),
    ]
    url = BASE + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "AgadirOpenSedimentMobility/1.0"})
    with urlopen(request, timeout=120) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
    if "tiff" not in content_type.lower() or not data.startswith((b"II*\x00", b"MM\x00*")):
        raise RuntimeError(f"EMODnet returned an unexpected response ({content_type}); no output was written.")
    out.write_bytes(data)
    manifest = {
        "source": "EMODnet Bathymetry WCS",
        "coverage": cfg["emodnet_coverage"],
        "bbox_wgs84": box,
        "request_url": url,
        "bytes": len(data),
        "vertical_datum_note": "Check source metadata for vertical reference and compilation details before fusion.",
    }
    (ROOT / "Data/Manifests/emodnet_bathymetry_download.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded regional EMODnet subset: {out} ({len(data) / 1e6:.2f} MB)")
    print("This grid is a comparison surface. Do not infer local vertical-datum equivalence from the file alone.")


if __name__ == "__main__":
    main()
