"""Assign a train/val ``split`` column to the filament chirality catalog.

Run from the repo root:

    python downstream_apps/filament_kyle/data/make_splits.py

Rewrites ``full_fil_data.csv`` in place: rows and labels are left alone, only the
``split`` column is added or replaced.

Deduplication is done by hand upstream, so this script does not remove anything
-- it *checks* the property instead. A filament survives on the disk for days, so
the catalog can record the same structure several times. Two rows in the same
hemisphere with the same chirality and no more than ``EPISODE_GAP`` apart are
near-duplicate views of one object: keeping both inflates the apparent sample
size, and splitting them across train/val leaks the answer. If such a pair
reappears, this raises rather than quietly splitting it.

The split is stratified on hemisphere x chirality. The catalog is close to
balanced in those two, and the Martin's Rule baseline reads that balance -- on 34
events an unstratified draw can easily hand val a lopsided class ratio and make
the baseline number meaningless.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).parent / "full_fil_data.csv"

EPISODE_GAP = pd.Timedelta("3d")  # two rows closer than this are one filament
VAL_FRACTION = 0.30
SEED = 42

STRATA = ["hemisphere", "chirality"]


def load_catalog() -> pd.DataFrame:
    """Read the catalog, dropping spreadsheet leftovers and validating labels."""
    df = pd.read_csv(CSV).drop(columns=["split"], errors="ignore")

    # Clearing cells in a spreadsheet without deleting the rows leaves ",," lines,
    # which pandas reads as all-NaN rows and which turn the integer label columns
    # into float64.
    blank = df.isna().all(axis=1)
    if blank.any():
        print(f"dropped {int(blank.sum())} blank row(s) from {CSV.name}")
        df = df.loc[~blank].copy()

    if df.isna().any().any():
        bad = df.loc[df.isna().any(axis=1)]
        raise ValueError(f"partially empty row(s):\n{bad}")

    df["ts"] = pd.to_datetime(df["start_time"], format="%Y-%m-%dT%H:%M:%S")
    if df["start_time"].duplicated().any():
        dups = df.loc[df["start_time"].duplicated(keep=False), "start_time"].tolist()
        raise ValueError(f"duplicate start_time values: {dups}")

    for col in STRATA:
        df[col] = df[col].astype(int)  # undo the float64 the NaN rows forced
    if not df["chirality"].isin([0, 1]).all():
        raise ValueError("chirality must contain only 0 or 1")

    return df.sort_values("ts").reset_index(drop=True)


def check_no_repeat_observations(df: pd.DataFrame) -> None:
    """Raise if any two same-label rows are close enough to be one filament."""
    offenders = []
    for _, rows in df.groupby(STRATA, sort=True):
        rows = rows.sort_values("ts")
        gaps = rows["ts"].diff()
        for i, gap in gaps.items():
            if pd.notna(gap) and gap <= EPISODE_GAP:
                prev = rows["start_time"].shift().loc[i]
                offenders.append(f"{prev} <-> {rows['start_time'].loc[i]} ({gap})")

    if offenders:
        raise ValueError(
            "rows within "
            f"{EPISODE_GAP} of each other share a hemisphere and chirality, so they "
            "are probably one filament observed twice. Drop one of each pair, or "
            "lower EPISODE_GAP if they really are distinct:\n  "
            + "\n  ".join(offenders)
        )

    closest = min(
        rows.sort_values("ts")["ts"].diff().min()
        for _, rows in df.groupby(STRATA, sort=True)
        if len(rows) > 1
    )
    print(f"no repeat observations within {EPISODE_GAP}; closest same-label pair is {closest}")


def assign_split(df: pd.DataFrame) -> pd.Series:
    """Label ~VAL_FRACTION of each stratum 'val', the rest 'train'."""
    rng = np.random.default_rng(SEED)
    split = pd.Series("train", index=df.index)

    for stratum, rows in df.groupby(STRATA, sort=True):
        n_val = max(1, round(VAL_FRACTION * len(rows)))
        split.loc[rng.permutation(rows.index.to_numpy())[:n_val]] = "val"
        print(f"  stratum hem={stratum[0]} chir={stratum[1]}: "
              f"{len(rows)} events -> {n_val} val")

    return split


def main() -> None:
    df = load_catalog()
    print(f"{CSV.name}: {len(df)} filaments\n")

    check_no_repeat_observations(df)

    print("\nassigning splits:")
    df["split"] = assign_split(df)

    out = df.drop(columns=["ts"])
    out.to_csv(CSV, index=False)

    n_train = int((out.split == "train").sum())
    n_val = int((out.split == "val").sum())
    print(f"\nwrote {CSV.name}: {n_train} train / {n_val} val "
          f"({n_val / len(out):.0%} val)")
    print(pd.crosstab([out["split"], out["hemisphere"]], out["chirality"]))


if __name__ == "__main__":
    main()
