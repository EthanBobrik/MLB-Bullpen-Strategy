"""
Bullpen form & fatigue features.

Given a pitch-level Statcast table, compute rolling/EWMA performance metrics for
each pitcher. These can be sampled at PA start times.

Key metrics:
- strike_ewma_h10: EWMA of strikes (swinging or in play)
- zone_ewma_h10: EWMA of pitches in the zone
- release_speed_delta: release_speed minus pitcher's historical mean
- release_spin_delta: release_spin_rate minus pitcher's historical mean
- strike30_shrunk: Beta-shrunk strike% over last 30 pitches
- last_outing_pitches, days_since_used, b2b_flag: fatigue proxies
"""
import numpy as np
import pandas as pd

def ewma(series:pd.Series, halflife: float) -> pd.Series:
    """Exponentially weighted moving average with half-life parameter"""
    alpha = np.log(2) / halflife
    return series.ewm(alpha=alpha, adjust=False).mean()

def add_pitcher_form_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """
    Add form & fatigue features at the pitch level per pitcher.

    You will typically use this, then sample rows at PA start to get form
    snapshot for (game_pk, pitcher) at decision time.

    Parameters
    ----------
    pitches : pd.DataFrame
        Raw Statcast with at least: game_pk, game_date, pitcher, pitch_number,
        description, release_speed, release_spin_rate, zone, type.

    Returns
    -------
    pd.DataFrame
        Same as input, but with added columns for each pitcher:
        - strike, in_zone
        - zone_ewma_h10, strike_ewma_h10
        - release_speed_delta, release_spin_delta
        - strike30_shrunk
        - last_outing_pitches, days_since_used, b2b_flag (approximated)
    """
    df = pitches.copy()
    df.sort_values(
        ["pitcher", "game_date", "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"],
        inplace=True,
    )

    # basic flags 
    df["strike_flag"] = df["type"].isin(["S", "X"]).astype(int)
    df["in_zone"] = (
        pd.to_numeric(df["zone"], errors="coerce")
          .between(1, 9)
          .fillna(False)
          .astype(int)
    )

    # Per-pitcher expanding baselines for speed/spin
    df["release_speed_mean"] = df.groupby("pitcher")["release_speed"].expanding().mean().reset_index(level=0, drop=True)
    df["release_spin_mean"] = df.groupby("pitcher")["release_spin_rate"].expanding().mean().reset_index(level=0, drop=True)

    df["release_speed_delta"] = df["release_speed"] - df["release_speed_mean"]
    df["release_spin_delta"] = df["release_spin_rate"] - df["release_spin_mean"]

    # Rolling/EWMA per pitcher
    def _apply_rolling(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        # EWMA half-life 10 pitches
        g["zone_ewma_h10"] = ewma(g["in_zone"].astype(float), halflife=10)
        g["strike_ewma_h10"] = ewma(g["strike_flag"].astype(float), halflife=10)

        # Rolling 30-pitch strike% with Beta(20,10) prior
        g["strike_30_sum"] = g["strike_flag"].rolling(30, min_periods=5).sum()
        g["strike_30_n"] = g["strike_flag"].rolling(30, min_periods=5).count()
        a0, b0 = 20, 10
        g["strike30_shrunk"] = (g["strike_30_sum"] + a0) / (g["strike_30_n"] + a0 + b0)
        return g

    df = df.groupby("pitcher", group_keys=False).apply(_apply_rolling)

    ## Approximate appearance-level fatigue:
    # Treat each (game_pk,pitcher) as an appearance.
    df['pitches_in_game'] = df.groupby(['pitcher','game_pk']).cumcount()+1

    # For last_outing_pitches and days_since_used, we need one row per (pitcher, game_pk)
    game_summary = (
        df.groupby(['pitcher','game_pk'], as_index=False)
        .agg({"pitches_in_game": "max", 'game_date': 'first'})
        .rename(columns={'pitches_in_game': 'pitches_this_game'})
    )
    game_summary.sort_values(['pitcher', 'game_date'], inplace=True)

    # Last outing workload (previous game's pitches)
    game_summary['last_outing_pitches'] = (
        game_summary.groupby('pitcher')['pitches_this_game'].shift(1).fillna(0)
    )

    # Use Statcast's pitcher_days_since_prev_game if available; otherwise fall back
    # to computing days between game dates.
    if 'pitcher_days_since_prev_game' in df.columns:
        ds = (
            df.groupby(['pitcher', 'game_pk'], as_index=False)['pitcher_days_since_prev_game']
              .first()
        )
        game_summary = game_summary.merge(ds, on=['pitcher', 'game_pk'], how='left')
        game_summary['days_since_used'] = game_summary['pitcher_days_since_prev_game'].fillna(99)
        game_summary.drop(columns=['pitcher_days_since_prev_game'], inplace=True)
    else:
        game_summary['prev_game_date'] = game_summary.groupby('pitcher')['game_date'].shift(1)
        game_summary['days_since_used'] = (
            pd.to_datetime(game_summary['game_date'])
            - pd.to_datetime(game_summary['prev_game_date'])
        ).dt.days.fillna(99)

    game_summary['b2b_flag'] = (game_summary['days_since_used'] == 1).astype(int)
    # Merge back onto pitch-level rows
    df = df.merge(
        game_summary[["pitcher", "game_pk", "last_outing_pitches", "days_since_used", "b2b_flag"]],
        on=["pitcher", "game_pk"],
        how="left",
    )

    return df