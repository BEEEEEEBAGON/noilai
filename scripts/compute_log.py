#!/usr/bin/env python
"""Append and read the GPU-hours log that answers checklist item C1.

The ARR Responsible NLP checklist asks for the total computational budget (GPU hours) and
the hardware; the plan asks that the log be kept from day one. This module keeps a plain
CSV with one row per session or run:

    date, platform, gpu_type, n_gpus, hours, run_id, purpose

`hours` is wall-clock time of the session or run; `n_gpus` the number of devices it held
(2 for a Kaggle 2xT4 session, 1 for a TPU v5e-8 session, 0 for API-only work). Device-hours
(hours x n_gpus) are what C1 reports; wall hours are what Kaggle's weekly quota counts.

    python scripts/compute_log.py append --platform kaggle --gpu t4 --n-gpus 2 --hours 1.75 \
        --run-id E1_main__gemma-3-1b-it --purpose E1_main
    python scripts/compute_log.py totals            # C1 summary, by device type and by purpose
    python scripts/compute_log.py show              # the raw rows

The default log lives at data/compute_log.csv (committed; data/runs/ is git-ignored). The
Kaggle notebooks write to the clone's copy and ship it with the run outputs; merge it into
the repository's copy when the results come back.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "data" / "compute_log.csv"
COLUMNS = ("date", "platform", "gpu_type", "n_gpus", "hours", "run_id", "purpose")
PLATFORMS = ("kaggle", "colab", "modal", "local", "api", "other")
# device type -> accounting kind; anything unknown is counted as a GPU
DEVICE_KINDS = {
    "t4": "gpu", "2xt4": "gpu", "p100": "gpu", "l4": "gpu", "a100": "gpu", "h100": "gpu",
    "tpu-v5e-8": "tpu", "tpu": "tpu",
    "cpu": "cpu", "api": "api", "none": "api",
}


@dataclass(frozen=True)
class Entry:
    date: str          # ISO date (YYYY-MM-DD)
    platform: str
    gpu_type: str
    n_gpus: int
    hours: float
    run_id: str
    purpose: str

    def __post_init__(self):
        dt.date.fromisoformat(self.date)                      # raises ValueError on a bad date
        if self.platform not in PLATFORMS:
            raise ValueError(f"platform {self.platform!r} not in {PLATFORMS}")
        if int(self.n_gpus) < 0 or int(self.n_gpus) != self.n_gpus:
            raise ValueError("n_gpus must be a non-negative integer")
        if float(self.hours) < 0:
            raise ValueError("hours must be non-negative")
        if not self.run_id:
            raise ValueError("run_id is required")

    @property
    def device_hours(self) -> float:
        return float(self.hours) * int(self.n_gpus)

    @property
    def kind(self) -> str:
        return DEVICE_KINDS.get(self.gpu_type.lower(), "gpu")

    def row(self) -> dict:
        d = asdict(self)
        d["n_gpus"] = int(self.n_gpus)
        d["hours"] = f"{float(self.hours):.4f}"
        return d


def today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def hours_since(t0: float) -> float:
    """Wall hours elapsed since a time.time() stamp (for the notebooks' run loops)."""
    return max(0.0, (time.time() - t0) / 3600.0)


def append_entry(path: Path, entry: Entry) -> Path:
    """Append one row, writing the header first when the file does not exist yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(entry.row())
    return path


def read_entries(path: Path) -> list[Entry]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[Entry] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for row in r:
            out.append(Entry(date=row["date"], platform=row["platform"], gpu_type=row["gpu_type"],
                             n_gpus=int(row["n_gpus"]), hours=float(row["hours"]),
                             run_id=row["run_id"], purpose=row["purpose"]))
    return out


def _bucket(entries: Iterable[Entry], key) -> dict[str, dict]:
    b: dict[str, dict] = {}
    for e in entries:
        k = key(e)
        d = b.setdefault(k, {"sessions": 0, "wall_hours": 0.0, "device_hours": 0.0})
        d["sessions"] += 1
        d["wall_hours"] += float(e.hours)
        d["device_hours"] += e.device_hours
    return dict(sorted(b.items()))


def totals(entries: list[Entry]) -> dict:
    """C1 summary: device-hours by kind (gpu/tpu), by device type, platform and purpose."""
    by_kind = _bucket(entries, lambda e: e.kind)
    return {
        "n_entries": len(entries),
        "wall_hours": sum(float(e.hours) for e in entries),
        "device_hours": sum(e.device_hours for e in entries),
        "gpu_hours": by_kind.get("gpu", {}).get("device_hours", 0.0),
        "tpu_hours": by_kind.get("tpu", {}).get("device_hours", 0.0),
        "by_gpu_type": _bucket(entries, lambda e: e.gpu_type.lower()),
        "by_platform": _bucket(entries, lambda e: e.platform),
        "by_purpose": _bucket(entries, lambda e: e.purpose),
        "first_date": min((e.date for e in entries), default=None),
        "last_date": max((e.date for e in entries), default=None),
    }


def format_totals(t: dict) -> str:
    lines = [f"compute log: {t['n_entries']} entries"
             + (f", {t['first_date']} .. {t['last_date']}" if t["first_date"] else "")]
    lines.append(f"  GPU device-hours: {t['gpu_hours']:.2f}    TPU device-hours: {t['tpu_hours']:.2f}"
                 f"    wall hours (quota): {t['wall_hours']:.2f}")
    for title, key in (("by device type", "by_gpu_type"), ("by platform", "by_platform"), ("by purpose", "by_purpose")):
        lines.append(f"  {title}:")
        for k, d in t[key].items():
            lines.append(f"    {k:<32} sessions {d['sessions']:>3}   wall {d['wall_hours']:>8.2f} h   device {d['device_hours']:>8.2f} h")
    lines.append("  checklist C1: report the GPU and TPU device-hours above with the device types.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("append", help="append one row")
    a.add_argument("--date", default=None, help="ISO date; default today (UTC)")
    a.add_argument("--platform", required=True, choices=PLATFORMS)
    a.add_argument("--gpu", required=True, help="device type: t4, p100, l4, tpu-v5e-8, api, ...")
    a.add_argument("--n-gpus", type=int, required=True)
    a.add_argument("--hours", type=float, required=True)
    a.add_argument("--run-id", required=True)
    a.add_argument("--purpose", required=True)
    sub.add_parser("totals", help="print the C1 summary")
    sub.add_parser("show", help="print the rows")
    args = ap.parse_args(argv)

    if args.cmd == "append":
        e = Entry(date=args.date or today(), platform=args.platform, gpu_type=args.gpu, n_gpus=args.n_gpus,
                  hours=args.hours, run_id=args.run_id, purpose=args.purpose)
        p = append_entry(args.log, e)
        print(f"[log ] {p}: {e.row()}")
        return 0
    entries = read_entries(args.log)
    if args.cmd == "show":
        w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        w.writeheader()
        for e in entries:
            w.writerow(e.row())
        return 0
    print(format_totals(totals(entries)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
