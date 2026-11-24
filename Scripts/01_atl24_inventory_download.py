from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import earthaccess
import h5py
import requests
from earthaccess.auth import Auth
from earthaccess.exceptions import LoginAttemptFailure, LoginStrategyUnavailable
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")
REQUIRED_PHOTON_FIELDS = ("class_ph", "lat_ph", "lon_ph", "ortho_h")


def expected_size_bytes(granule) -> int | None:
    """Read the CMR granule size when available (ATL24 commonly reports MB with unit NA)."""
    umm = granule.get("umm", {})
    entries = (umm.get("DataGranule", {}) or {}).get("ArchiveAndDistributionInformation", []) or []
    if not entries:
        return None
    try:
        size = float(entries[0]["Size"])
    except (KeyError, TypeError, ValueError):
        return None
    unit = str(entries[0].get("SizeUnit", "")).strip().upper()
    factors = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    # CMR currently returns ATL24 sizes in MiB even when its SizeUnit is "NA";
    # this was checked against complete granules downloaded from this inventory.
    factor = factors.get(unit, 1024**2 if unit in ("", "NA") else None)
    return int(size * factor) if factor else None


def valid_atl24_hdf5(path: Path) -> bool:
    """Check HDF5 structure and readable endpoints before treating a file as complete."""
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with h5py.File(path, "r") as h5:
            for beam in BEAMS:
                if beam not in h5 or not all(key in h5[beam] for key in REQUIRED_PHOTON_FIELDS):
                    continue
                datasets = [h5[beam][key] for key in REQUIRED_PHOTON_FIELDS]
                if any(ds.ndim != 1 for ds in datasets):
                    continue
                lengths = [len(ds) for ds in datasets]
                if len(set(lengths)) != 1:
                    continue
                if lengths[0]:
                    for ds in datasets:
                        _ = ds[0]
                        _ = ds[-1]
                return True
    except (OSError, RuntimeError, ValueError):
        return False
    return False


def complete_granule(path: Path, granule) -> bool:
    if not valid_atl24_hdf5(path):
        return False
    expected = expected_size_bytes(granule)
    if expected is not None:
        tolerance = max(1024**2, int(expected * 0.01))
        if path.stat().st_size < expected - tolerance:
            return False
    return True


def download_one(granule, target: Path, auth: Auth, progress: tqdm, progress_lock: threading.Lock):
    links = granule.data_links()
    if not links:
        return "failed", "(granule without a data link)"
    url = links[0]
    filename = Path(urlparse(url).path).name
    final_path = target / filename
    expected = expected_size_bytes(granule)
    temporary_path = final_path.with_suffix(final_path.suffix + ".part")

    for attempt in range(1, 5):
        transferred = 0
        offset = temporary_path.stat().st_size if temporary_path.exists() else 0
        if expected is not None and offset:
            tolerance = max(1024**2, int(expected * 0.01))
            if offset >= expected - tolerance:
                if valid_atl24_hdf5(temporary_path):
                    temporary_path.replace(final_path)
                    return "downloaded", filename
                with progress_lock:
                    progress.update(-offset)
                temporary_path.unlink(missing_ok=True)
                offset = 0
        if expected is not None and offset > expected:
            with progress_lock:
                progress.update(-offset)
            temporary_path.unlink(missing_ok=True)
            offset = 0
        session = auth.get_session()
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 120),
            ) as response:
                if response.status_code == 416:
                    if expected is not None and offset >= expected - max(1024**2, int(expected * 0.01)) and valid_atl24_hdf5(temporary_path):
                        temporary_path.replace(final_path)
                        return "downloaded", filename
                    with progress_lock:
                        if offset:
                            progress.update(-offset)
                    temporary_path.unlink(missing_ok=True)
                    raise IOError(f"Range request not satisfiable for {filename}; restarting this granule")
                response.raise_for_status()
                mode = "wb"
                if offset and response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        with progress_lock:
                            progress.update(-offset)
                        temporary_path.unlink(missing_ok=True)
                        raise IOError(
                            f"Server returned an unexpected Content-Range for {filename}: {content_range}"
                        )
                    mode = "ab"
                elif offset and response.status_code == 200:
                    # The server ignored Range; restart this attempt cleanly.
                    with progress_lock:
                        progress.update(-offset)
                    offset = 0
                with temporary_path.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
                        transferred += len(chunk)
                        with progress_lock:
                            progress.update(len(chunk))

            actual = temporary_path.stat().st_size
            if expected is not None:
                tolerance = max(1024**2, int(expected * 0.01))
                if actual < expected - tolerance:
                    raise IOError(f"Short transfer: {actual} of about {expected} bytes")
            if not valid_atl24_hdf5(temporary_path):
                raise IOError("Downloaded file failed the ATL24 HDF5 structure check")
            temporary_path.replace(final_path)
            return "downloaded", filename
        except (requests.exceptions.RequestException, OSError, RuntimeError, ValueError) as exc:
            with progress_lock:
                tqdm.write(f"Attempt {attempt}/4 failed for {filename}: {type(exc).__name__}: {exc}")
            if attempt == 4:
                return "failed", filename
            time.sleep(min(2 ** (attempt - 1), 8))
        finally:
            session.close()
    return "failed", filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and optionally download open NASA ICESat-2 ATL24 v003 granules for the study AOI.")
    parser.add_argument("--download", action="store_true", help="Authenticate with NASA Earthdata and download the inventoried granules.")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    box = cfg["aoi_bbox_wgs84"]
    granules = earthaccess.search_data(
        short_name="ATL24",
        version=cfg["atl24_version"],
        bounding_box=(box["west"], box["south"], box["east"], box["north"]),
        temporal=(cfg["date_start"], cfg["date_end"]),
    )
    out_csv = ROOT / "Data/Manifests/atl24_v003_inventory.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for granule in granules:
        meta = granule.get("umm", {})
        title = meta.get("GranuleUR") or granule.get("meta", {}).get("native-id", "")
        temporal = meta.get("TemporalExtent", {}).get("RangeDateTime", {})
        start = temporal.get("BeginningDateTime", "")
        end = temporal.get("EndingDateTime", "")
        links = granule.data_links()
        rows.append({"granule": title, "start_utc": start, "end_utc": end, "data_url": links[0] if links else ""})
    with out_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["granule", "start_utc", "end_utc", "data_url"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"ATL24 V{cfg['atl24_version']} granules intersecting the preliminary AOI: {len(rows)}")
    print(f"Inventory: {out_csv}")
    print("The AOI intersection is a catalogue result, not a count of usable seabed photons.")
    if not args.download:
        print("No files were downloaded. Review the granule inventory, then rerun with --download.")
        return
    target = ROOT / "Data/Raw/ATL24"
    target.mkdir(parents=True, exist_ok=True)

    # Authenticate without constructing earthaccess.Store: its extra /profile
    # request can fail independently after valid credentials have been accepted.
    auth = Auth()
    try:
        try:
            auth.login(strategy="netrc", persist=False)
        except (LoginStrategyUnavailable, LoginAttemptFailure):
            auth.login(strategy="interactive", persist=True)
    except requests.exceptions.ConnectionError as exc:
        raise SystemExit(
            "Earthdata authentication was interrupted by a network connection error. "
            "No ATL24 granules were downloaded; wait for the connection to recover and rerun."
        ) from exc

    # earthaccess skips any existing filename, including zero-byte or truncated
    # files left by Ctrl+C. Validate the existing local copy first so reruns resume
    # by retaining complete granules and redownloading incomplete ones.
    removed_partials = []
    retained_complete = 0
    for granule in granules:
        links = granule.data_links()
        if not links:
            continue
        filename = Path(urlparse(links[0]).path).name
        local_file = target / filename
        if not local_file.exists():
            continue
        if complete_granule(local_file, granule):
            retained_complete += 1
            continue
        size = local_file.stat().st_size if local_file.is_file() else 0
        local_file.unlink()
        removed_partials.append({"file": filename, "bytes_removed": size})
        print(f"Removed incomplete local file; it will be downloaded again: {filename}")

    audit_dir = ROOT / "Results/QC"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "atl24_resume_cleanup.csv"
    with audit_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["checked_utc", "file", "bytes_removed"])
        writer.writeheader()
        for row in removed_partials:
            writer.writerow({"checked_utc": datetime.now(timezone.utc).isoformat(), **row})
    print(f"Complete local granules retained: {retained_complete}; incomplete files queued for retry: {len(removed_partials)}")

    # Show byte-level transfer progress directly in the terminal.
    incomplete_granules = []
    total_expected = 0
    initial_bytes = 0
    for granule in granules:
        links = granule.data_links()
        if not links:
            incomplete_granules.append(granule)
            continue
        local_file = target / Path(urlparse(links[0]).path).name
        expected = expected_size_bytes(granule)
        if expected:
            total_expected += expected
        if complete_granule(local_file, granule):
            initial_bytes += min(local_file.stat().st_size, expected or local_file.stat().st_size)
        else:
            partial_path = local_file.with_suffix(local_file.suffix + ".part")
            if partial_path.is_file():
                initial_bytes += min(partial_path.stat().st_size, expected or partial_path.stat().st_size)
            incomplete_granules.append(granule)

    results = []
    if incomplete_granules:
        progress_lock = threading.Lock()
        with tqdm(
            total=total_expected or None,
            initial=initial_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="ATL24 transfer",
            mininterval=0.5,
        ) as progress:
            with ThreadPoolExecutor(max_workers=min(3, len(incomplete_granules))) as pool:
                futures = [
                    pool.submit(download_one, granule, target, auth, progress, progress_lock)
                    for granule in incomplete_granules
                ]
                for future in as_completed(futures):
                    status, filename = future.result()
                    results.append((status, filename))
                    tqdm.write(f"{status}: {filename}")
    incomplete = []
    for granule in granules:
        links = granule.data_links()
        if not links:
            incomplete.append("(granule without a data link)")
            continue
        filename = Path(urlparse(links[0]).path).name
        if not complete_granule(target / filename, granule):
            incomplete.append(filename)
    print(f"Downloaded or retained complete granules: {len(granules) - len(incomplete)}/{len(granules)} to {target}")
    if incomplete:
        print(f"Incomplete or missing granules: {len(incomplete)}. Rerun this same command later to resume them.")
        print("Downloaded fragments are retained as .part files and resumed with HTTP Range requests.")
    else:
        print("All inventoried ATL24 granules passed the local HDF5 and size checks.")
    print("Check each granule’s photon confidence, class, spatial distribution, and EGM08 vertical reference before analysis.")


if __name__ == "__main__":
    main()
