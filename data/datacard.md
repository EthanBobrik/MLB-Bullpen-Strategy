1) Data Sources (pybaseball only)

Functions called:
	•	pybaseball.statcast(start_dt, end_dt, team=None) — pitch-level table (the backbone).
	•	pybaseball.statcast_pitcher(start_dt, end_dt, player_id) — reliever appearances to compute form & fatigue.
	•	pybaseball.statcast_batter(start_dt, end_dt, player_id) — batter pitch histories (optional for granular features).
	•	pybaseball.schedule_and_record(season, team) — per-game calendar (bootstrap blocks).
	•	pybaseball.playerid_lookup(last, first) / playerid_reverse_lookup(ids) — map names ⇄ IDs.
	•	pybaseball.pitching_stats(season, qual=0) and batting_stats(season, qual=0) — season priors for Bayesian shrinkage.

We infer lineups and batting order from plate-appearance sequence in the Statcast pitch table (robust in practice).
