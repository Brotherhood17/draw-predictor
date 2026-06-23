"""
Feature engineering shared by train_model.py and predict.py.

Given a dataframe of played matches (chronologically ordered), builds the
per-match features the model uses:

  - rolling form (points per game, goals scored/conceded) for each team
    over their last N games, computed using ONLY information available
    before that match (no leakage)
  - table position / points gap between the two teams at that point
  - head-to-head draw rate between the two teams
  - bookmaker-implied draw probability (if odds columns are present)

predict.py reuses build_team_history() to get each team's current rolling
stats, then build_fixture_features() to score upcoming fixtures with the
same feature definitions the model was trained on.
"""

import pandas as pd
import numpy as np

FORM_WINDOW = 5  # games of recent form to look back on


def load_results_csv(path):
    """Loads a football-data.co.uk-style CSV (or our synthetic one)."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _result_points(goals_for, goals_against):
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def build_team_history(df):
    """
    Walks through matches in chronological order and builds a long-format
    table of every team's results, so we can compute rolling form and
    cumulative table standing as-of any date without leaking future data.
    """
    records = []
    for _, row in df.iterrows():
        records.append({
            "Date": row["Date"], "Team": row["HomeTeam"], "Opponent": row["AwayTeam"],
            "GF": row["FTHG"], "GA": row["FTAG"], "Venue": "H",
            "Points": _result_points(row["FTHG"], row["FTAG"]),
        })
        records.append({
            "Date": row["Date"], "Team": row["AwayTeam"], "Opponent": row["HomeTeam"],
            "GF": row["FTAG"], "GA": row["FTHG"], "Venue": "A",
            "Points": _result_points(row["FTAG"], row["FTHG"]),
        })
    history = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    return history


def _stats_before(history, team, as_of_date, window=FORM_WINDOW):
    """Recent form + season-to-date cumulative stats for `team` strictly
    before `as_of_date`."""
    past = history[(history["Team"] == team) & (history["Date"] < as_of_date)]
    if past.empty:
        return {
            "ppg_recent": 1.0, "gf_recent": 1.0, "ga_recent": 1.0,
            "season_points": 0.0, "season_gd": 0.0, "games_played": 0,
        }
    recent = past.tail(window)
    season = past[past["Date"] >= (as_of_date - pd.Timedelta(days=300))]
    return {
        "ppg_recent": recent["Points"].mean(),
        "gf_recent": recent["GF"].mean(),
        "ga_recent": recent["GA"].mean(),
        "season_points": season["Points"].sum(),
        "season_gd": (season["GF"] - season["GA"]).sum(),
        "games_played": len(season),
    }


def _h2h_draw_rate(df, home, away, as_of_date):
    past = df[
        (df["Date"] < as_of_date) &
        (((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
         ((df["HomeTeam"] == away) & (df["AwayTeam"] == home)))
    ]
    if past.empty:
        return 0.25  # league-average prior if no history
    return (past["FTR"] == "D").mean()


def build_fixture_features(df, history, home, away, match_date):
    """Builds one feature row for a (home, away) fixture at match_date,
    using only information available before that date."""
    h = _stats_before(history, home, match_date)
    a = _stats_before(history, away, match_date)

    features = {
        "ppg_gap": abs(h["ppg_recent"] - a["ppg_recent"]),
        "gf_gap": abs(h["gf_recent"] - a["gf_recent"]),
        "ga_gap": abs(h["ga_recent"] - a["ga_recent"]),
        "combined_goal_expectation": h["gf_recent"] + a["gf_recent"],
        "season_points_gap": abs(h["season_points"] - a["season_points"]),
        "season_gd_gap": abs(h["season_gd"] - a["season_gd"]),
        "home_ppg_recent": h["ppg_recent"],
        "away_ppg_recent": a["ppg_recent"],
        "h2h_draw_rate": _h2h_draw_rate(df, home, away, match_date),
        "min_games_played": min(h["games_played"], a["games_played"]),
    }
    return features


def build_training_table(df):
    """Builds the full feature matrix + labels for every match in df."""
    history = build_team_history(df)
    rows = []
    for _, m in df.iterrows():
        feats = build_fixture_features(df, history, m["HomeTeam"], m["AwayTeam"], m["Date"])
        feats["is_draw"] = 1 if m["FTR"] == "D" else 0

        # bookmaker-implied draw probability, if odds are present
        if "B365D" in df.columns and pd.notna(m.get("B365D")):
            implied = []
            for col in ["B365H", "B365D", "B365A"]:
                if pd.notna(m.get(col)) and m[col] > 0:
                    implied.append(1.0 / m[col])
            total = sum(implied) if implied else 0
            if total > 0 and "B365D" in df.columns and m["B365D"] > 0:
                feats["bookmaker_draw_prob"] = (1.0 / m["B365D"]) / total
            else:
                feats["bookmaker_draw_prob"] = np.nan
        rows.append(feats)

    table = pd.DataFrame(rows)
    # drop the earliest games where teams have almost no history (noisy signal)
    table = table[table["min_games_played"] >= 3].reset_index(drop=True)
    return table
