"""
Scores upcoming fixtures with the trained model and builds the 3 release
tiers for matchday:

    Tier 1 ("The Locks")      - top 3 by predicted draw probability
    Tier 2 ("The Watch List") - next 2
    Tier 3 ("Final Calls")    - next 3

Usage:
    python scripts/predict.py --tier 1
    python scripts/predict.py --tier 2
    python scripts/predict.py --tier 3

Each run scores ALL upcoming fixtures fresh, then reveals one more tier's
worth of picks and writes/updates public/predictions.json - merging with
whatever tiers were already revealed earlier in the day, so running tier 2
doesn't erase tier 1.

Swap get_upcoming_fixtures() below for a real API call (football-data.org,
API-Football, etc.) when you're ready to go live - see README.md.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd

from features import load_results_csv, build_team_history, build_fixture_features
from analysis import build_analysis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "draw_model.pkl")
PREDICTIONS_PATH = os.path.join(ROOT, "public", "predictions.json")
SAMPLE_FIXTURES_PATH = os.path.join(ROOT, "data", "sample_fixtures.json")
RAW_DIR = os.path.join(ROOT, "data", "raw")

TIER_CONFIG = {
    1: {"label": "The Locks", "count": 3},
    2: {"label": "The Watch List", "count": 2},
    3: {"label": "Final Calls", "count": 3},
}


def get_upcoming_fixtures():
    """
    Returns a list of {"home": ..., "away": ..., "date": "YYYY-MM-DD"}.

    TODO: replace this with a real call to football-data.org or
    API-Football using the FOOTBALL_API_KEY env var. Falling back to the
    bundled sample fixtures for now so the pipeline runs end-to-end.
    """
    api_key = os.environ.get("FOOTBALL_API_KEY")
    if api_key:
        # Example shape for football-data.org - fill in your competition IDs
        # and uncomment:
        #
        # import requests
        # resp = requests.get(
        #     "https://api.football-data.org/v4/matches",
        #     headers={"X-Auth-Token": api_key},
        #     params={"status": "SCHEDULED"},
        # )
        # data = resp.json()
        # return [
        #     {"home": m["homeTeam"]["name"], "away": m["awayTeam"]["name"],
        #      "date": m["utcDate"][:10]}
        #     for m in data["matches"]
        # ]
        pass

    with open(SAMPLE_FIXTURES_PATH) as f:
        return json.load(f)


def load_history_for_features():
    import glob
    paths = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    frames = [load_results_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    history = build_team_history(df)
    return df, history


def score_fixtures(fixtures):
    bundle = joblib.load(MODEL_PATH)
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    df, history = load_history_for_features()
    league_draw_rate = (df["FTR"] == "D").mean()

    scored = []
    for fx in fixtures:
        match_date = pd.to_datetime(fx["date"])
        feats = build_fixture_features(df, history, fx["home"], fx["away"], match_date)
        # bookmaker_draw_prob isn't available pre-match unless you wire in live
        # odds; fill with the same league-average prior used during training
        # for any rows missing it.
        row = {col: feats.get(col, 0.25) for col in feature_cols}
        X = pd.DataFrame([row])[feature_cols]
        draw_prob = model.predict_proba(X)[0, 1]

        analysis = build_analysis(feats, draw_prob)

        scored.append({
            "home": fx["home"],
            "away": fx["away"],
            "date": fx["date"],
            "draw_probability": round(float(draw_prob), 3),
            "vs_baseline": round(float(draw_prob) - league_draw_rate, 3),
            "confidence": analysis["confidence"],
            "analysis": analysis["factors"],
            "supporting_stats": {
                "recent_form_gap_ppg": round(float(feats.get("ppg_gap", 0)), 2),
                "table_points_gap": round(float(feats.get("season_points_gap", 0)), 1),
                "combined_recent_goals": round(float(feats.get("combined_goal_expectation", 0)), 2),
                "head_to_head_draw_rate": round(float(feats.get("h2h_draw_rate", 0)), 2),
            },
        })

    scored.sort(key=lambda m: m["draw_probability"], reverse=True)
    return scored, league_draw_rate


def assign_tiers(scored, up_to_tier):
    """Splits the sorted fixture list into tiers 1..up_to_tier."""
    result = {}
    idx = 0
    for tier_num in range(1, up_to_tier + 1):
        cfg = TIER_CONFIG[tier_num]
        picks = scored[idx: idx + cfg["count"]]
        idx += cfg["count"]
        result[str(tier_num)] = {
            "label": cfg["label"],
            "released_at": datetime.now(timezone.utc).isoformat(),
            "picks": picks,
        }
    return result


def update_predictions_file(new_tiers, league_draw_rate=None):
    existing = {}
    if os.path.exists(PREDICTIONS_PATH):
        with open(PREDICTIONS_PATH) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    tiers = existing.get("tiers", {})
    tiers.update(new_tiers)  # overwrite/add only the tiers we just computed

    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "league_draw_rate": round(float(league_draw_rate), 3) if league_draw_rate is not None else existing.get("league_draw_rate"),
        "tiers": tiers,
    }
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {PREDICTIONS_PATH}")


def main(tier_num):
    fixtures = get_upcoming_fixtures()
    print(f"Scoring {len(fixtures)} upcoming fixtures...")
    scored, league_draw_rate = score_fixtures(fixtures)

    for m in scored:
        print(f"  {m['draw_probability']:.1%} ({m['confidence']})  {m['home']} vs {m['away']} ({m['date']})")
        for factor in m["analysis"]:
            print(f"      - {factor}")

    tiers = assign_tiers(scored, up_to_tier=tier_num)
    update_predictions_file(tiers, league_draw_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, required=True, choices=[1, 2, 3],
                         help="Reveal up through this tier (1=just Tier 1, 3=all three tiers)")
    args = parser.parse_args()
    main(args.tier)
