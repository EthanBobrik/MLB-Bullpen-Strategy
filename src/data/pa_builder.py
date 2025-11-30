"""
Build plate appearances (PAs) and decision checkpoints from pitch-level Statcast.

Responsibilities:
 - Group pitches into PAs
 - Derive base state, outs, scores at PA start
 - Attach pitcher_on_mound, batter, home/away teams
 - Compute basic fatigue: pitch_count up to PA, times_through_order (TTO)
"""
import pandas as pd
import numpy as np

def _compute_base_state(row: pd.Series) -> int:
    """
    Convert on_1b/on_2b/on_3b into a bit-encoded integer 0-7.

    Bits (LSB first): [1B, 2B, 3B]
    """
    b1 = int(pd.notna(row.get("on_1b")))
    b2 = int(pd.notna(row.get("on_2b")))
    b3 = int(pd.notna(row.get("on_3b")))
    return (b1 << 0) | (b2 << 1) | (b3 << 2)

def build_pas(pitches: pd.DataFrame) -> pd.DataFrame:
    """
    Build a plate-appearance level table from pitch-level Statcast.

    Parameters
    ----------
    pitches : pd.DataFrame
        Raw Statcast with at least: game_pk, game_date, inning, inning_topbot,
        at_bat_number, pitch_number, batter, pitcher, home_team, away_team,
        outs_when_up, on_1b/2b/3b.

    Returns
    -------
    pd.DataFrame
        One row per PA start, with all columns.
    """
    df = pitches.copy()
    df.sort_values(
        ['game_pk','game_date','inning','inning_topbot','at_bat_number','pitch_number'],
        inplace=True
    )
    # Encode half inning as 0=Top 1=Bot
    df['half'] = df['inning_topbot'].map({'Top':0,'Bot':1})
    #Unique PA key
    df['pa_key'] = (
        df['game_pk'].astype(str)
        + "_"
        + df['inning'].astype(str)
        + "_"
        + df['half'].astype(str)
        + "_"
        + df['at_bat_number'].fillna(-1).astype(int).astype(str)
    )

    # First pitch in each PA is the start-of-PA snapshot
    grp = df.groupby('pa_key',sort=False)
    pa = grp.first().reset_index(drop=True)

    pa['base_state'] = pa.apply(_compute_base_state, axis=1)
    pa.rename(columns={
        'outs_when_up' : 'outs',
        'pitcher' : 'pitcher_on_mound'
    }, inplace=True)

    # Determine batting/fielding team:
    # if half == 0 (Top), batting = away_team, fielding = home_team; vice versa
    pa['batting_team'] = np.where(pa['half'] == 0, pa['away_team'],pa['home_team'])
    pa['fielding_team'] = np.where(pa['half'] == 0, pa['home_team'],pa['away_team'])

    if "bat_score_diff" in pa.columns:
        pa["score_diff"] = pa["bat_score_diff"]
    else:
        pa["score_diff"] = pa["bat_score"] - pa["fld_score"]

    # Handedness and matchup features
    # Batter handedness one-hot encodings (if available)
    if "stand" in pa.columns:
        pa["batter_is_left"] = (pa["stand"] == "L").astype(int)
        pa["batter_is_right"] = (pa["stand"] == "R").astype(int)
        pa["batter_is_switch"] = (pa["stand"] == "S").astype(int)
    else:
        pa["batter_is_left"] = 0
        pa["batter_is_right"] = 0
        pa["batter_is_switch"] = 0

    # Pitcher handedness one-hot encodings (if available)
    if "p_throws" in pa.columns:
        pa["pitcher_is_left"] = (pa["p_throws"] == "L").astype(int)
        pa["pitcher_is_right"] = (pa["p_throws"] == "R").astype(int)
    else:
        pa["pitcher_is_left"] = 0
        pa["pitcher_is_right"] = 0

    # Platoon advantage flag: 1 if batter has the platoon advantage vs current pitcher
    # For switch-hitters, treat as always having platoon advantage when pitcher throws L/R.
    if "stand" in pa.columns and "p_throws" in pa.columns:
        pa["is_platoon_advantage"] = np.where(
            pa["stand"] == "S",
            1,
            (pa["stand"] != pa["p_throws"]).astype(int),
        )
    else:
        pa["is_platoon_advantage"] = 0

    return pa.copy()

def attach_pitcher_fatigue_features(pa:pd.DataFrame, pitches: pd.DataFrame) -> pd.DataFrame:
    """
    Attach pitch_count and times_through_order (TTO) features to the PA table.

    pitch_count: total pitches thrown by current pitcher in this game before PA start.
    TTO: how many previous PAs this batter has faced this pitcher in this game.

    Parameters
    ----------
    pa : pd.DataFrame
        Output from build_pas().
    pitches : pd.DataFrame
        Raw Statcast pitch-level table.

    Returns
    -------
    pd.DataFrame
        PA table with added columns:
        - pitch_count
        - tto (int)
    """
    df = pitches.copy()
    df.sort_values(
        ["game_pk", "game_date", "inning", "inning_topbot", "at_bat_number", "pitch_number"],
        inplace=True,
    )

    # Add numeric half, like build_pas
    df["half"] = df["inning_topbot"].map({"Top": 0, "Bot": 1})

    # Cumulative pitch count per (game_pk, pitcher)
    df['pitch_count_cum'] = df.groupby(['game_pk','pitcher']).cumcount() + 1

    # We'll merge PA rows with the *last* pitch index of previous PA to get counts at PA start.
    # For simplicity, approximate pitch_count at PA start as min pitch_count_cum within that PA.
    first_pitches = (
        df.groupby(
            ["game_pk", "inning", "half", "at_bat_number", "pitcher"], as_index=False
        )
        .agg({"pitch_count_cum": "min", "n_priorpa_thisgame_player_at_bat": "first"})
        .rename(columns={"pitcher": "pitcher_on_mound"})
    )

    merged = pa.merge(
        first_pitches[[
            "game_pk", "inning", "half", "at_bat_number", "pitcher_on_mound", "pitch_count_cum", "n_priorpa_thisgame_player_at_bat"
        ]],
        on = ["game_pk", "inning", "half", "at_bat_number", "pitcher_on_mound"],
        how='left'
    )
    merged.rename(columns={"pitch_count_cum": "pitch_count"},inplace=True)

    # Use Statcast's n_priorpa_thisgame_player_at_bat as our TTO signal:
    # it already encodes how many prior PAs this batter has had in this game.
    if "n_priorpa_thisgame_player_at_bat" in merged.columns:
        merged["tto"] = merged["n_priorpa_thisgame_player_at_bat"].fillna(0).astype(int)
        merged.drop(columns=["n_priorpa_thisgame_player_at_bat"], inplace=True)
    else:
        # Fallback for older data without this column: approximate via cumcount.
        merged.sort_values(["game_pk", "pitcher_on_mound", "batter", "at_bat_number"], inplace=True)
        merged["tto"] = (
            merged.groupby(["game_pk", "pitcher_on_mound", "batter"]).cumcount()
        )

    return merged.sort_values(["game_pk", "inning", "half", "at_bat_number"]).reset_index(drop=True)
