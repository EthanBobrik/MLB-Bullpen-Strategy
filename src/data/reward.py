"""
Run Expectancy (RE)–based reward utilities for bullpen RL using Statcast.

This module is intentionally SIMPLE and uses the Statcast columns:

    - delta_run_exp
        The change in Run Expectancy before the PITCH and after the PITCH
        (from the batting team's perspective).

We do three main things:

1. add_pa_delta_run_exp:
   - Aggregate pitch-level delta_run_exp to PA-level, producing
     'delta_re_pa' = change in Run Expectancy over the entire plate appearance.

2. annotate_terminals:
   - Mark which PA rows are the last in a half-inning and game so that
     we can stop folding rewards correctly.

3. fold_smdp_reward:
   - Given a sequence of PA states with 'delta_re_pa', fold the per-PA
     RE changes into a single SMDP-style reward over a fixed horizon
     (e.g. 3 batters) to model the MLB three-batter minimum.

From the FIELDING team perspective, reducing expected runs is good.
Since delta_run_exp is defined for the BATTING team, our per-PA reward
for the fielding team is approximately:

    reward_pa = -delta_re_pa

i.e. we negate the run expectancy change.
"""

from typing import Tuple
import numpy as np
import pandas as pd

def add_pa_delta_run_exp(
        pa: pd.DataFrame, 
        pitches:pd.DataFrame,
        delta_col: str = 'delta_run_exp',
        out_col: str = 'delta_re_pa') -> pd.DataFrame:
    """
    Aggregate Statcast pitch-level delta_run_exp to a PA-level column.

    Statcast definitions:
    - delta_run_exp: change in Run Expectancy before the PITCH and after
      the PITCH, from the batting team's perspective.

    At the PA level, we want:

        delta_re_pa = RE_after_last_pitch_of_PA - RE_before_first_pitch_of_PA

    which is (approximately) the sum of delta_run_exp over all pitches
    in that PA.

    Parameters
    ----------
    pa : pd.DataFrame
        PA-level table from build_pas(...), containing at least:
          ['game_pk', 'at_bat_number'].
    pitches : pd.DataFrame
        Full Statcast pitch-level DataFrame containing at least:
          ['game_pk', 'at_bat_number', delta_col].
    delta_col : str, default 'delta_run_exp'
        Name of the pitch-level column in pitches.
    out_col : str, default 'delta_re_pa'
        Name of the PA-level output column.

    Returns
    -------
    pd.DataFrame
        Copy of pa with a new column out_col, where each value is the
        sum of delta_run_exp across all pitches in that PA. Missing
        values are filled with 0.0.
    """
    df_pa = pa.copy()

    # Sum delta_run_exp over pitches in each PA
    # Note: At the pitch level, delta_run_exp is from batting perspective.
    pitch_delta = (
        pitches.groupby(["game_pk", "at_bat_number"], as_index=False)[delta_col]
        .sum()
        .rename(columns={delta_col: out_col})
    )

    # Merge onto PA table
    df_pa = df_pa.merge(
        pitch_delta,
        on=["game_pk", "at_bat_number"],
        how="left",
    )

    df_pa[out_col] = df_pa[out_col].fillna(0.0).astype(np.float32)

    return df_pa

def annotate_terminals(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'half_inning_over' and 'game_over' boolean columns to the PA table.

    This is needed so that when we fold rewards over a 3-batter window,
    we stop if the half-inning or game ends before we hit the full horizon.

    Parameters
    ----------
    pa : pd.DataFrame
        PA-level table, sorted by:
        ['game_pk', 'inning', 'half', 'at_bat_number'].

    Returns
    -------
    pd.DataFrame
        Copy of pa with two new boolean columns:
          - 'half_inning_over': True on the last PA of each (game_pk, inning, half)
          - 'game_over': True on the last PA of each game_pk
    """
    df = pa.copy()
    # Half-inning terminal: last PA in each (game, inning, half)
    df["half_inning_over"] = False
    last_indices = (
        df.groupby(["game_pk", "inning", "half"], sort=False).tail(1).index
    )
    df.loc[last_indices, "half_inning_over"] = True

    #game terminal: last PA in each game
    df['game_over'] = False
    last_game_indices = df.groupby('game_pk',sort=False).tail(1).index
    df.loc[last_game_indices,'game_over'] = True

    return df

def fold_smdp_reward(
        pa_seq: pd.DataFrame,
        start_idx:int,
        gamma:float = 0.99,
        penalty_pull:float = 0.005,
        pulled:bool = False,
        max_horizon: int = 3, # min number of batters in an inning
        delta_re_col:str = 'delta_re_pa'
) -> Tuple[float, int, bool]:
    """
    Fold SMDP reward using Statcast delta_run_exp aggregated per PA.

    We assume:
    - pa_seq is a *local* PA sequence (e.g. all PAs for one
      (game_pk, fielding_team, half)), sorted in chronological order.
    - pa_seq has columns:
        - delta_re_col      : change in Run Expectancy for batting team
                              over this PA (delta_re_pa).
        - 'half_inning_over': bool, True if this PA is the last in the half.
        - 'game_over'       : bool, True if this PA is the last in the game.

    Reward is defined from the FIELDING team's perspective:

        reward_pa_t ≈ - delta_re_pa_t

      since delta_re_pa_t is defined from the batting team's perspective.

    We then fold over a finite horizon (e.g. 3 batters) to respect the
    three-batter minimum:

        R = sum_{k=0}^{K-1} gamma^k * [ - delta_re_pa_{t+k} ]
            - 1[pulled] * penalty_pull

    where we roll forward until:
      - we've taken max_horizon steps (default 3), or
      - the half-inning ends, or
      - the game ends, or
      - we run out of PAs in pa_seq.

    Parameters
    ----------
    pa_seq : pd.DataFrame
        Local PA sequence with delta_re_col, 'half_inning_over', 'game_over'.
    start_idx : int
        Index *within pa_seq* at which the decision is made.
    gamma : float, default 0.99
        Discount factor applied to future per-PA rewards.
    penalty_pull : float, default 0.005
        Small penalty subtracted from reward when pulled=True to discourage
        frivolous bullpen changes.
    pulled : bool, default False
        Whether the action at start_idx was "pull the current pitcher".
    max_horizon : int, default 3
        Maximum number of PA transitions to fold over (3 for the MLB rule).
    delta_re_col : str, default "delta_re_pa"
        Name of the PA-level run-expectancy delta column in pa_seq.

    Returns
    -------
    reward : float
        Folded, discounted reward for this decision (fielding perspective).
    next_idx : int
        Index within pa_seq where the *next* decision may occur (end of folded window).
    done : bool
        True if the game is over at next_idx (terminal state).
    """
    n = len(pa_seq)

    # If this is the last PA in the sequence, there is no future to roll over.
    if start_idx >= n - 1:
        done_flag = bool(pa_seq.iloc[start_idx].get("game_over", False))
        reward = -penalty_pull if pulled else 0.0
        return float(reward), int(start_idx), done_flag

    R = 0.0
    steps = 0
    curr_idx = start_idx

    while True:
        # Reward for this PA (fielding perspective)
        delta_re_pa = float(pa_seq.iloc[curr_idx][delta_re_col])
        R += (gamma ** steps) * (-delta_re_pa)

        steps += 1
        half_over = bool(pa_seq.iloc[curr_idx].get("half_inning_over", False))
        game_over = bool(pa_seq.iloc[curr_idx].get("game_over", False))

        if steps >= max_horizon or half_over or game_over or curr_idx >= n - 1:
            break

        curr_idx += 1

    # Apply small cost for using a bullpen move
    if pulled:
        R -= float(penalty_pull)

    done_flag = bool(pa_seq.iloc[curr_idx].get("game_over", False))
    return float(R), int(curr_idx), done_flag