#!/usr/bin/env python3
"""Stream compatible per-geometry Schur CSV files into one training dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference_header: list[str] | None = None
    rows_written = 0
    geometry_counts: dict[int, int] = {}

    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        for path in args.inputs:
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.reader(source)
                header = next(reader)
                if reference_header is None:
                    reference_header = header
                    if "geometry_id" not in header:
                        raise ValueError(f"{path} has no geometry_id column")
                    writer.writerow(header)
                elif header != reference_header:
                    raise ValueError(
                        f"CSV schema mismatch in {path}; generate every input with "
                        "the same --PaddingLocal and --PaddingSkeleton values"
                    )

                geometry_column = header.index("geometry_id")
                for row in reader:
                    writer.writerow(row)
                    geometry_id = int(float(row[geometry_column]))
                    geometry_counts[geometry_id] = geometry_counts.get(geometry_id, 0) + 1
                    rows_written += 1

    print(f"Wrote {rows_written} rows to {args.output}")
    print("Rows by geometry_id: " + ", ".join(
        f"{key}={geometry_counts[key]}" for key in sorted(geometry_counts)
    ))


if __name__ == "__main__":
    main()
