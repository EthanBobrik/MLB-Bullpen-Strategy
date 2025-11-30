"""
Bullpen availability masks.

Determines which relievers are eligible to enter at a given decision, based on:
- not already used in this game
- rest days above a threshold
- last outing pitch count below a threshold

This module expects the per-pitcher form snapshot to follow the naming conventions from bullpen_form.py,
in particular 'days_since_used' and 'last_outing_pitches', and does not create any new overlapping fields.
"""

from typing import Iterable, List, Mapping

import numpy as np
import pandas as pd

def build_availability(
        team_relievers: Iterable[int],
        form_snapshot: Mapping[int, pd.Series],
        already_used_ids: Iterable[int],
        rest_min_days: int = 1,
        last_outing_max: int = 35
) -> np.ndarray:
    """
    Build a boolean availability mask for a team's bullpen at one decision point.

    Parameters
    ----------
    team_relievers : iterable of int
        MLBAM IDs (or internal IDs) for bullpen pitchers on this team.
    form_snapshot : mapping[int, pd.Series]
        Mapping from reliever_id -> per-reliever form row *as of decision time*,
        containing at least columns ['days_since_used', 'last_outing_pitches'].
    already_used_ids : iterable of int
        Relievers that have already appeared in this game.
    rest_min_days : int
        Minimum days since last appearance to be considered available.
    last_outing_max : int
        Maximum pitches in last outing to be considered available.

    Returns
    -------
    np.ndarray
        Boolean mask of shape [R], in the same order as team_relievers.
    """
    used = set(already_used_ids)
    mask: List[bool] = []
    
    for rid in team_relievers:
        if rid in used:
            mask.append(False)
            continue

        row = form_snapshot.get(rid,None)
        if row is None:
            # if we don't have form info (e.g. rookie), assume available
            mask.append(True)
            continue

        days_since = row.get('days_since_used', 99)
        last_p = row.get('last_outing_pitches', 0)

        # Normalize NaN to defaults and ensure numeric types
        if pd.isna(days_since):
            days_since = 99
        if pd.isna(last_p):
            last_p = 0

        days_since = float(days_since)
        last_p = float(last_p)

        ok_rest = days_since >= rest_min_days
        ok_last = last_p <= last_outing_max

        mask.append(bool(ok_last and ok_rest))
    
    return np.array(mask, dtype=bool)