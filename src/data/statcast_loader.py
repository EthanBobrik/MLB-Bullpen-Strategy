"""
Utilities to download and cache Statcast data using pybaseball.

This module is responsible ONLY for:
- talking to pybaseball.statcast
- saving monthly Parquet files under data/raw/
- loading & concatenating those Parquet files

Everything downstream should work from Parquet, not from live API calls.
"""

try:
    from pybaseball import statcast
except ModuleNotFoundError:
    statcast = None

import pandas as pd
from pathlib import Path
from typing import Iterable,List, Optional

from pathlib import Path

# Resolve project root as two levels up from this file: src/data -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def pull_month(year: int,month:int ,force:bool = False) -> Path:
    """
    Download one month of Statcast data and save to Parquet.

    Parameters
    ----------
    year : int
        Season year, e.g. 2023.
    month : int
        Month number 1–12.
    force : bool
        If False and file already exists, skip download.

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    RAW_DIR.mkdir(parents=True,exist_ok=True)
    out_path = RAW_DIR / f"statcast_{year}_{month:02d}.parquet"

    if out_path.exists() and not force:
        print(f"[statcast_loader] Using cached {out_path}")
        return out_path

    start = pd.Timestamp(year,month,1).strftime('%Y-%m-%d')
    end = (pd.Timestamp(year,month,1) + pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')

    print(f"[statcast_loader] Pulling Statcast {start} -> {end} ...")
    df = statcast(start,end)
    df.to_parquet(out_path)
    print(f"[statcast_loader] Saved {len(df):,} rows to {out_path}")
    return out_path

def pull_season_by_month(years: List[int], months: Optional[Iterable[int]] = None, force:bool = False) -> List[Path]:
    """
    Download an entire season month-by-month.

    Parameters
    ----------
    year : List[int]
        Season years.
    months : iterable of int, optional
        Subset of months to download; defaults to 3–10 (typical MLB).
    force : bool
        Force re-download even if files exist.

    Returns
    -------
    list[Path]
        Paths of all monthly Parquet files.
    """
    if months is None:
        months = range(3,11)

    paths: List[Path] = []
    for year in years:
        for m in months:
            try:
                paths.append(pull_month(year,m,force=force))
            except Exception as e:
                print(f"[statcast_loader] Failed for {year}-{m:02d}: {e}")
    return paths

def load_raw_statcast(years: List[int]) -> pd.DataFrame:
    """
    Load all raw Statcast Parquets for the given years into a single DataFrame.

    Parameters
    ----------
    years : List[int]

    Returns
    -------
    pd.DataFrame
        Concatenated pitch-level Statcast for the years.
    """
    all_dfs = []

    for year in years:
        pattern = f"statcast_{year}_"
        files = sorted(p for p in RAW_DIR.glob("statcast_*.parquet") if pattern in p.name)
        if not files:
            raise FileNotFoundError(f"No statcast*.parquet files for year {year} in {RAW_DIR}")

        dfs = [pd.read_parquet(p) for p in files]
        df_year = pd.concat(dfs, ignore_index=True)
        all_dfs.append(df_year)

    if not all_dfs:
        raise FileNotFoundError(f"No statcast parquet files found in {RAW_DIR} for years={years}")

    df = pd.concat(all_dfs, ignore_index=True)
    df.sort_values(
        ["game_pk", "game_date", "inning", "inning_topbot", "at_bat_number", "pitch_number"],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df