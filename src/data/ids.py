"""
ID utilities.

Thin wrappers atound pybaseball's player Id and team ID helpers
Provides:
- stable MLBAM player IDs
- bats/throws (handedness) info
- team ID mappings
"""

from functools import lru_cache
from typing import Optional

import pandas as pd
from pybaseball import playerid_lookup, playerid_reverse_lookup, team_ids

@lru_cache(maxsize=None)
def get_team_id_map() -> pd.DataFrame:
    """
    Return the full team ID table from pybaseball.team_ids().

    Returns
    -------
    pd.DataFrame
        Columns typically include: team, team_id, mlb_abbrev, etc.
    """
    return team_ids()

def lookup_player_id(last: str, first:str) -> Optional[int]:
    """
    Resolve a player's MLBAM ID using (last, first) name.

    Parameters
    ----------
    last : str
    first : str

    Returns
    -------
    int or None
        MLBAM ID if a match is found, else None.
    """
    df = playerid_lookup(last,first)
    if df.empty:
        return None
    # If multiple rows, you might want to disambiguate by team, active status, etc.
    return int(df.iloc[0]['key_mlbam'])

@lru_cache(maxsize=None)
def reverse_lookup_mlbam(mlbam_id:int) -> pd.Series:
    """
    Reverse lookup for a player, given MLBAM ID.

    Returns
    -------
    pd.Series
        Contains name, bats, throws, etc.
    """
    df = playerid_reverse_lookup([mlbam_id],key_type='mlbam')
    if df.empty:
        raise ValueError(f'No reverse lookup results for MLBAM {mlbam_id}')
    return df.iloc[0]