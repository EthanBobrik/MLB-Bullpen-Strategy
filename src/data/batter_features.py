"""
Batter feature construction from Statcast pitch-level data.

This module builds:
- A per-batter feature table using only Statcast columns.
- A lineup-window tensor [B, H, d_hit] aligned with pa['next_hitters_ids'].

It assumes:
- Pitches come from pybaseball.statcast() (the same DataFrame you pass into build_pas).
- PAs come from src.data.pa_builder.build_pas(pitches) and a separate lineup builder
  will add 'next_hitters_ids' to that PA table.
"""
import numpy as np
import pandas as pd
from typing import Iterable, Optional

def _build_pitch_flags(pitches:pd.DataFrame) -> pd.DataFrame:
    """
    Add basic boolean flags and xwOBA proxy to the raw Statcast pitch table.

    Uses only Statcast columns:
      - description
      - type (B/S/X)
      - zone
      - launch_speed
      - launch_speed_angle
      - estimated_woba_using_speedangle
      - woba_value

    Returns
    -------
    pd.DataFrame
        Same as input, with extra columns:
        is_ball, is_strike_called, is_strike_swinging, is_strike_foul,
        is_in_play, is_swing, is_whiff, is_contact, is_in_zone,
        is_hard_hit, is_barrel, xwoba_est
    """
    df = pitches.copy()

    desc = df['description'].fillna("").str.lower()
    ptype = df['type'].fillna('')

    df['is_ball'] = (ptype == 'B').astype(int)
    df['is_strike_called'] = desc.str.contains('called_strike').astype(int)
    df['is_strike_swinging'] = desc.str.contains('swinging_strike').astype(int)
    df['is_strike_foul'] = desc.str.contains('foul').astype(int)
    df['is_in_play'] = (ptype == 'X').astype(int)

    df['is_swing'] = (
        df['is_strike_swinging']
        | df['is_strike_foul']
        | df['is_in_play']
        | desc.str.contains('foul_tip')
        | desc.str.contains('hit_into_play')
    ).astype(int)

    df['is_whiff'] = df['is_strike_swinging'].astype(int)
    df['is_contact'] = (df['is_in_play'] | df['is_strike_foul']).astype(int)
    df['is_in_zone'] = (
    pd.to_numeric(df['zone'], errors='coerce')
      .between(1, 9, inclusive="both")
      .fillna(False)
      .astype(int)
    )
    df["is_hard_hit"] = (
        pd.to_numeric(df["launch_speed"], errors="coerce")
        .ge(95)
        .fillna(False)
        .astype(int)
    )

    df["is_barrel"] = (
        pd.to_numeric(df["launch_angle"], errors="coerce")
        .eq(6)
        .fillna(False)
        .astype(int)
    )          
    df['xwoba_est'] = df['estimated_woba_using_speedangle'].fillna(df['woba_value'])

    return df

def build_batter_feature_table(
        pitches: pd.DataFrame,
        min_pa: int =20
) -> pd.DataFrame:
    """
    Construct a batter-level feature table using Statcast pitch-level data only.

    Features include:
    - pa_count: number of PAs
    - pitch_count: number of pitches seen
    - swing_rate: swings / pitches
    - whiff_rate: whiffs / swings
    - contact_rate: contacts / swings
    - chase_rate: swings at out-of-zone pitches / pitches
    - z_swing_rate: swings at in-zone pitches / in-zone pitches
    - z_contact_rate: contacts on in-zone swings / in-zone swings
    - ball_rate: balls / pitches
    - called_strike_rate: called strikes / pitches
    - hard_hit_rate: hard-hit balls / contact events
    - barrel_rate: barrels / contact events
    - xwoba_mean: mean xwOBA proxy per pitch
    - bb_rate_pa: walk+HBP per PA (approx)
    - k_rate_pa: strikeouts per PA (approx)
    - handed_L, handed_R, handed_S: fraction of pitches (or PA-ending pitches) where the batter's recorded stand was L/R/S

    Parameters
    ----------
    pitches : pd.DataFrame
        Full Statcast pitch-level table for a season (same one passed to build_pas).
        Must contain at least: batter, description, type, zone, stand,
        launch_speed, launch_speed_angle, estimated_woba_using_speedangle,
        woba_value, game_pk, at_bat_number, pitch_number, events.
    min_pa : int
        Minimum plate appearances to keep a batter.

    Returns
    -------
    pd.DataFrame
        Indexed by 'batter' (MLBAM id) with feature columns.
    """

    df = _build_pitch_flags(pitches)

    # Pitch level aggreagates per batter
    grp_pitch = df.groupby('batter',as_index=False)
    pitch_agg = grp_pitch.agg(
        pitch_count=('batter','count'),
        swing_count=('is_swing','sum'),
        whiff_count = ('is_whiff','sum'),
        contact_count = ('is_contact','sum'),
        ball_count = ('is_ball','sum'),
        called_strike_count = ('is_strike_called','sum'),
        in_zone_count = ('is_in_zone','sum'),
        hard_hit_count = ('is_hard_hit','sum'),
        barrel_count = ('is_barrel','sum'),
        xwoba_sum = ('xwoba_est','sum'),
    )

    # Zone-specific aggregates
    df_in_zone = df[df['is_in_zone'] == 1]
    grp_zone = df_in_zone.groupby('batter',as_index=False)
    zone_agg = grp_zone.agg(
        z_pitch_count = ('batter','count'),
        z_swing_count = ('is_swing','sum'),
        z_contact_count = ('is_contact','sum'),
    )

    feat = pitch_agg.merge(zone_agg, on="batter", how="left").fillna(0.0)

    # PA level aggreagates for BB% and K%
    df_sorted = df.sort_values(['game_pk','at_bat_number','pitch_number'])
    pa_last = df_sorted.groupby(['game_pk','at_bat_number'],as_index=False).last()

    ev = pa_last['events'].fillna('').str.lower()
    pa_last['is_bb'] = ev.isin(['walk','hit_by_pitch']).astype(int)
    pa_last['is_k'] = ev.str.contains('strikeout').astype(int)

    grp_pa = pa_last.groupby('batter', as_index=True)
    pa_agg = grp_pa.agg(
        pa_count=('batter','count'),
        bb_count=('is_bb','sum'),
        k_count=('is_k','sum'),
    ).reset_index()

    feat = feat.merge(pa_agg, on="batter", how="left").fillna(0.0)

    # Handedness features per batter (based on Statcast 'stand' column)
    if "stand" in df.columns:
        stand_agg = df.groupby("batter", as_index=False).agg(
            handed_L=("stand", lambda s: float((s == "L").mean()) if len(s) > 0 else 0.0),
            handed_R=("stand", lambda s: float((s == "R").mean()) if len(s) > 0 else 0.0),
            handed_S=("stand", lambda s: float((s == "S").mean()) if len(s) > 0 else 0.0),
        )
        feat = feat.merge(stand_agg, on="batter", how="left").fillna(0.0)
    else:
        # Fallback: no handedness information available
        feat["handed_L"] = 0.0
        feat["handed_R"] = 0.0
        feat["handed_S"] = 0.0

    # Rate features
    eps = 1e-9

    feat['swing_rate'] = feat['swing_count']/(feat['pitch_count']+eps)
    feat['whiff_rate'] = feat['whiff_count']/(feat['swing_count']+eps)
    feat['contact_rate'] = feat['contact_count']/(feat['swing_count']+eps)

    feat['ball_rate'] = feat['ball_count'] / (feat['pitch_count']+eps)
    feat['called_strike_rate'] = feat['called_strike_count']/(feat['pitch_count']+eps)

    feat['z_swing_rate'] = feat['z_swing_count']/(feat['z_pitch_count']+eps)
    feat['z_contact_rate'] = feat['z_contact_count']/(feat['z_swing_count']+eps)
    
    feat['hard_hit_rate'] = feat['hard_hit_count']/(feat['contact_count']+eps)
    feat['barrel_rate'] = feat['barrel_count']/(feat['contact_count']+eps)

    feat['xwoba_mean'] = feat['xwoba_sum']/(feat['pitch_count']+eps)
    
    feat['bb_rate_pa'] = feat['bb_count']/(feat['pa_count']+eps)
    feat['k_rate_pa'] = feat['k_count']/(feat['pa_count']+eps)

    # Chase rate = swings out of zone pitches / pitches
    feat['o_swing_count'] = feat['swing_count'] - feat['z_swing_count']
    feat['chase_rate'] = feat['o_swing_count'] / (feat['pitch_count']+eps)

    #filter low-sample batters
    feat = feat[feat['pa_count'] >= min_pa].copy()

    feat.set_index("batter", inplace=True)

    # Select and order columns
    cols = [
        "pa_count",
        "pitch_count",
        "swing_rate",
        "whiff_rate",
        "contact_rate",
        "chase_rate",
        "z_swing_rate",
        "z_contact_rate",
        "ball_rate",
        "called_strike_rate",
        "hard_hit_rate",
        "barrel_rate",
        "xwoba_mean",
        "bb_rate_pa",
        "k_rate_pa",
        "handed_L",
        "handed_R",
        "handed_S",
    ]
    feat = feat[cols].astype(float)
    feat.index.name = "batter"

    return feat

def build_next_hitter_features(
        pa: pd.DataFrame,
        batter_features: pd.DataFrame,
        H: int,
        feature_cols: Optional[Iterable[str]] = None,
        default_zero: bool = True
) -> np.ndarray:
    """
    Convert pa['next_hitters_ids'] into an array [B, H, d_hit] using a
    precomputed batter feature table.

    Assumes:
    - pa comes from build_pas(pitches) + a lineup builder that added
      'next_hitters_ids' as list[int] (MLBAM batter IDs).

    Parameters
    ----------
    pa : pd.DataFrame
        Plate-appearance table with a column 'next_hitters_ids' (list[int]).
    batter_features : pd.DataFrame
        Indexed by batter ID (MLBAM) with feature columns from
        build_batter_feature_table().
    H : int
        Window size (# of upcoming hitters).
    feature_cols : iterable of str, optional
        Subset of columns from batter_features to use. If None, use all.
    default_zero : bool
        If True, batters not found in batter_features get a zero vector.
        If False, use the global mean feature vector instead.

    Returns
    -------
    np.ndarray
        Array of shape [B, H, d_hit], where B = len(pa),
        d_hit = len(feature_cols or batter_features.columns).
    """
    if feature_cols is None:
        feature_cols = list(batter_features.columns)
    feature_cols = list(feature_cols)

    bf = batter_features.copy()
    bf = bf[feature_cols].astype(float)

    bf_index = {
        int(bid): row.values.astype(np.float32)
        for bid, row in bf.iterrows()
    }

    B = len(pa)
    d_hit = len(feature_cols)
    X = np.zeros((B,H,d_hit), dtype=np.float32)

    if not default_zero and len(bf) >0:
        mean_vec = bf.mean(axis=0).to_numpy(dtype=np.float32)
    else:
        mean_vec = np.zeros(d_hit, dtype=np.float32)

    for i, hitters in enumerate(pa['next_hitters_ids']):
        hitters = list(hitters)
        #normalize length to H
        if len(hitters) <H:
            hitters = hitters + [None] *(H-len(hitters))
        else:
            hitters = hitters[:H]

        for j, bid in enumerate(hitters):
            if bid is None:
                X[i,j,:] = mean_vec if not default_zero else 0.0
            else:
                bid_int = int(bid)
                if bid_int in bf_index:
                    X[i,j,:] = bf_index[bid_int]
                else:
                    X[i,j,:] = mean_vec if not default_zero else 0.0
                    
    return X