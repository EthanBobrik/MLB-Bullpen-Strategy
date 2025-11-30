"""
Lineup window / pocket builder.

Given a PA table with [game_pk, batting_team, batter] in batting order
within each game, this module:

- infers a stable batting order per (game_pk, batting_team)
- records each PA's index in that order
- builds a window of the next H hitters (wrapping around)
"""
from typing import Dict, List
import numpy as np
import pandas as pd


def infer_batting_order(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Infer batting order index per (game_pk, batting_team) based on first appearance
    of each batter in that game for that team.

    Parameters
    ----------
    pa : pd.DataFrame
        Must contain: game_pk, batting_team, batter, inning, half, at_bat_number.
        Here `half` is the numeric encoding produced by build_pas:
          0 = Top, 1 = Bottom.

    Returns
    -------
    pd.DataFrame
        pa with an added 'lineup_idx' column (0-based order within team for that game).
    """
    pa = pa.copy()
    pa.sort_values(
        ["game_pk", "batting_team", "inning", "half", "at_bat_number"],
        inplace=True,
    )

    lineup_idx: List[int] = []

    for (_, team), gdf in pa.groupby(["game_pk", "batting_team"], sort=False):
        order: Dict[int, int] = {}
        next_idx = 0
        idxs: List[int] = []

        for _, row in gdf.iterrows():
            batter_id = int(row["batter"])
            if batter_id not in order:
                order[batter_id] = next_idx
                next_idx += 1
            idxs.append(order[batter_id])

        lineup_idx.extend(idxs)

    pa["lineup_idx"] = lineup_idx
    return pa


def add_lineup_window(pa: pd.DataFrame, H: int = 5) -> pd.DataFrame:
    """
    Add a lineup window of the next H hitters for each PA.

    Parameters
    ----------
    pa : pd.DataFrame
        Output from infer_batting_order().
        Must contain: game_pk, batting_team, inning, half, at_bat_number, batter.
    H : int
        Window length (number of upcoming hitters to encode).

    Returns
    -------
    pd.DataFrame
        pa with an added column 'next_hitters_ids' (list[int]) where each
        element is the list of the next H batters in real game order
        (wrapping around the lineup if needed).
    """
    pa = pa.copy()
    pa.sort_values(
        ["game_pk", "batting_team", "inning", "half", "at_bat_number"],
        inplace=True,
    )

    next_ids: List[List[int]] = []

    for (_, team), gdf in pa.groupby(["game_pk", "batting_team"], sort=False):
        batters = gdf["batter"].tolist()
        n = len(batters)
        # Map back from index labels to positional indices in this group
        idx_map = {idx: pos for pos, idx in enumerate(gdf.index)}

        for row_idx in gdf.index:
            pos = idx_map[row_idx]
            nxt = [batters[(pos + j) % n] for j in range(1, H + 1)]
            next_ids.append(nxt)

    pa["next_hitters_ids"] = next_ids
    return pa


def build_positional_encodings(num_positions: int, H: int) -> np.ndarray:
    """
    Simple positional encodings for the lineup window.

    For now we just encode 'distance from current PA' as one-hot over [1..H],
    which is cheap and works fine for most use cases.

    Parameters
    ----------
    num_positions : int
        Number of examples (B).
    H : int
        Window length.

    Returns
    -------
    np.ndarray
        Shape [B, H, H] (one-hot distance position).
    """
    B = num_positions
    pe = np.zeros((B, H, H), dtype=np.float32)
    for d in range(H):
        pe[:, d, d] = 1.0
    return pe