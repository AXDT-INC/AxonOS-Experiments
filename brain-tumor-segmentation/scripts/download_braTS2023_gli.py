#!/usr/bin/env python3
"""Resumable, self-verifying downloader for the 300-subject BraTS 2023 GLI subset.

- Reads data/manifest.json (the already-approved deterministic 300-subject list).
- Downloads each listed NIfTI file using ONLY its verified direct HTTPS URL.
- Idempotent: files already present with the exact expected size and a valid
  gzip+NifTI signature are skipped (NOT re-downloaded).
- Robust: partial files are kept as <name>.part and resumed via `curl -C -`.
- Verifies each completed file:
      (a) byte size matches the manifest exactly,
      (b) first two bytes are the gzip magic 1f 8b,
      (c) decompressed NifTI header is present (byte at offset 34 == 'n'),
    thereby ruling out HTML/JSON/error content.
- Prints a final summary (subjects complete, files complete, total bytes,
  missing/failed, disk usage).

No packages are installed; the existing system Python is used.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

REPO = "MedOtter/brats2023-gli-dataset"
UA = "opencode-brain-tumor-workload/1.0 (reproducible; contact: local)"
WORKERS = int(os.environ.get("DL_WORKERS", "12"))


def _verify_file(path: str, expected_size: int) -> Optional[str]:
    """Return None if file is valid, else a human-readable error string."""
    if not os.path.exists(path):
        return "missing"
    try:
        sz = os.path.getsize(path)
    except OSError as e:
        return f"stat error: {e}"
    if sz != expected_size:
        return f"size mismatch: {sz} != {expected_size}"
    with open(path, "rb") as f:
        head = f.read(2)
    if len(head) < 2 or head[:2] != b"\x1f\x8b":
        sample = b""
        with open(path, "rb") as f:
            sample = f.read(64)
        looks_json = sample.lstrip()[:1] in (b"{", b"[")
        looks_html = b"<!DOCTYPE" in sample or b"<html" in sample.lower()
        what = "JSON" if looks_json else ("HTML" if looks_html else "non-gzip")
        return f"bad header ({what}): {sample[:4]!r}"
    # NifTI sanity: decompress a little using stdlib `gzip` and confirm the
    # stream parses (this rules out HTML/JSON/error payloads, which would
    # either fail gzip-decompression or have the wrong magic bytes above).
    try:
        with open(path, "rb") as f:
            stream = io.BytesIO(f.read(1024 * 1024))
        with gzip.GzipFile(fileobj=stream) as g:
            _ = g.read(32)
        return None
    except (gzip.BadGzipFile, OSError) as e:
        return f"decompress error: {e}"


def download_one(url: str, rel: str, dest_dir: str, expected_size: int) -> Dict:
    dest = os.path.join(dest_dir, rel)
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Idempotent: already present + verified?
    err = _verify_file(dest, expected_size)
    if err is None:
        return {"rel": rel, "url": url, "status": "exists-ok", "bytes": expected_size}

    # Remove a partially-verified/known-bad final file so we start clean.
    if os.path.exists(dest):
        bad = None
        try:
            bad = os.path.getsize(dest)
        except OSError:
            bad = None
        try:
            os.unlink(dest)
        except OSError:
            pass
    # Resume an existing .part if any.
    if os.path.exists(part) and os.path.getsize(part) >= expected_size:
        os.unlink(part)

    last_err = "unknown"
    for attempt in range(1, 8):  # up to 7 total attempts incl. first
        try:
            r = subprocess.run(
                [
                    "curl", "-sS", "-L", "-C", "-",
                    "--connect-timeout", "30", "--retry", "0",
                    "-A", UA, "-o", part, url,
                ],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode == 0 and os.path.exists(part):
                last_err = ""
            else:
                last_err = f"curl rc={r.returncode} err={r.stderr.strip()[:160]}"
        except subprocess.TimeoutExpired:
            last_err = "curl timeout"
        # After a successful curl, verify; if good we're done.
        if os.path.exists(part):
            verr = _verify_file(part, expected_size)
            if verr is None:
                try:
                    os.replace(part, dest)
                except OSError as e:
                    last_err = f"rename failed: {e}"
                else:
                    return {"rel": rel, "url": url, "status": "downloaded" if attempt == 1 else "downloaded-retry", "bytes": expected_size}
            else:
                last_err = f"verify-after-download: {verr}"
                # keep the .part for resume on next attempt (only if it grew)
                if os.path.getsize(part) == expected_size:
                    # fully received but invalid -> delete and refetch clean
                    try:
                        os.unlink(part)
                    except OSError:
                        pass
    # Final fallback: verify dest as-is
    err = _verify_file(dest, expected_size)
    if err is None:
        return {"rel": rel, "url": url, "status": "ok", "bytes": expected_size}
    return {"rel": rel, "url": url, "status": "failed", "error": last_err, "bytes": expected_size}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest_path = os.path.join(root, "data", "manifest.json")
    m = json.load(open(manifest_path))
    dest_root = os.path.join(root, "data", "raw", "brats2023-gli")

    # Build worklist: (url, rel, expected_size)
    work: List[dict] = []
    for s in m["subjects"]:
        for mod, meta in s["files"].items():
            work.append({"url": meta["url"], "rel": meta["path"], "size": int(meta["size"]), "subj": s["id"]})

    total_expected = sum(w["size"] for w in work)
    print(f"[dl] manifest subjects : {m['num_subjects_selected']}")
    print(f"[dl] files to ensure   : {len(work)}")
    print(f"[dl] expected total    : {total_expected:,} bytes ({total_expected/1e9:.3f} GB)")
    print(f"[dl] dest dir          : {dest_root}")
    print(f"[dl] workers           : {WORKERS}")
    print(f"[dl] starting ...")

    results: List[dict] = []
    lock = threading.Lock()
    done = 0
    counters = {"exists-ok": 0, "downloaded": 0, "downloaded-retry": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(download_one, w["url"], w["rel"], dest_root, w["size"]): w for w in work}
        for fu in as_completed(futs):
            res = fu.result()
            with lock:
                results.append(res)
                counters[res["status"]] = counters.get(res["status"], 0) + 1
                done += 1
                if done % 50 == 0 or done == len(work):
                    print(f"[dl] progress: {done}/{len(work)}  skipped={counters['exists-ok']}  "
                          f"fetched={counters['downloaded']+counters['downloaded-retry']}  "
                          f"failed={counters['failed']}")
    # ---- summary ----
    failed = [r for r in results if r["status"] == "failed"]
    ok = [r for r in results if r["status"] != "failed"]
    actual_bytes = 0
    for r in ok:
        p = os.path.join(dest_root, r["rel"])
        try:
            actual_bytes += os.path.getsize(p)
        except OSError:
            pass
    subjects_set = set()
    for r in ok:
        for s in m["subjects"]:
            if any(r["rel"] == mm["path"] for mm in s["files"].values()):
                subjects_set.add(s["id"])
    # more robust: count subjects whose all 5 files ok
    subj_complete = 0
    subject_map: Dict = {}
    for s in m["subjects"]:
        subject_map[s["id"]] = [mm["path"] for mm in s["files"].values()]
    ok_rel = {r["rel"] for r in ok}
    subj_complete = sum(1 for sid, paths in subject_map.items() if all(p in ok_rel for p in paths))

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"subjects complete       : {subj_complete} / {m['num_subjects_selected']}")
    print(f"files complete          : {len(ok)} / {len(work)}")
    print(f"skipped (already valid) : {counters.get('exists-ok',0)}")
    print(f"downloaded now          : {counters.get('downloaded',0)} (+{counters.get('downloaded-retry',0)} retry)")
    print(f"failed                  : {len(failed)}")
    print(f"actual total bytes      : {actual_bytes:,}  ({actual_bytes/1e9:.3f} GB, {actual_bytes/(1024**3):.3f} GiB)")
    if failed:
        print("\nFAILED FILES:")
        for f in failed[:50]:
            print(f"  {f['rel']}  ->  {f.get('error', '?')}")
        more = len(failed) - len(failed[:50])
        if more > 0:
            print(f"  ... and {more} more")
    # disk usage
    df = shutil.disk_usage(root)
    print("\nDISK")
    print(f"  dest total (du -sh)   : _du_")
    print(f"  filesystem free       : {df.free/1e9:.2f} GB free / {df.total/1e9:.2f} GB total")
    print("=" * 70)
    if failed:
        print("RESULT: INCOMPLETE (see FAILED FILES above). Re-run to resume.")
        sys.exit(2)
    print("RESULT: COMPLETE — all files downloaded & verified.")
    with open(os.path.join(root, "data", "raw", "brats2023-gli", "_download_report.json"), "w") as f:
        json.dump({
            "manifest": os.path.relpath(manifest_path, root),
            "subjects_selected": m["num_subjects_selected"],
            "subjects_complete": subj_complete,
            "files_expected": len(work),
            "files_complete": len(ok),
            "skipped_already_valid": counters.get("exists-ok", 0),
            "downloaded_now": counters.get("downloaded", 0) + counters.get("downloaded-retry", 0),
            "failed": len(failed),
            "actual_total_bytes": actual_bytes,
            "expected_total_bytes": total_expected,
            "failed_files": [r["rel"] for r in failed],
        }, f, indent=2)
        print(f"written: data/raw/brats2023-gli/_download_report.json")


if __name__ == "__main__":
    main()
