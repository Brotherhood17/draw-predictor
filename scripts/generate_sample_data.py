"""
Generates a synthetic-but-realistic results dataset so you can run the
whole pipeline (features -> train -> predict) before plugging in real
football-data.co.uk data.

Each fake team has a hidden "strength" rating. Goals are simulated from
those ratings (Poisson), so matches between evenly-matched teams really do
draw more often - which gives the model genuine signal to learn, the same
shape of signal you'd see in real data (just cleaner/simpler).

Run: python scripts/generate_sample_data.py
"""

import csv
import os
import random
from datetime import date, timedelta

random.seed(42)

TEAMS = [f"Team {chr(65+i)}" for i in range(16)]  # Team A..P
# hidden strength ratings, roughly 0.8 - 2.2 expected goals scored vs an average side
STRENGTH = {t: round(random.uniform(0.8, 2.2), 2) for t in TEAMS}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sample_results.csv")


def simulate_goals(lam):
    """Simple Poisson sampler without numpy dependency."""
    L = pow(2.718281828, -lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def simulate_season(season_start_year, matchday_gap_days=4):
    rows = []
    current_date = date(season_start_year, 8, 10)
    teams = TEAMS[:]
    # round-robin home/away, twice (double round robin like a real league)
    fixtures = []
    for i, home in enumerate(teams):
        for away in teams:
            if home != away:
                fixtures.append((home, away))
    random.shuffle(fixtures)

    for home, away in fixtures:
        home_strength = STRENGTH[home]
        away_strength = STRENGTH[away]
        # home advantage bump
        home_lambda = max(0.3, home_strength * 1.1 - away_strength * 0.5 + 0.4)
        away_lambda = max(0.3, away_strength - home_strength * 0.35)

        home_goals = simulate_goals(home_lambda)
        away_goals = simulate_goals(away_lambda)

        if home_goals > away_goals:
            ftr = "H"
        elif home_goals < away_goals:
            ftr = "A"
        else:
            ftr = "D"

        # fake bookmaker odds, loosely consistent with strength gap
        gap = home_strength - away_strength
        draw_odds = round(max(2.6, 3.6 - abs(gap) * 0.5), 2)
        home_odds = round(max(1.3, 2.6 - gap * 0.4), 2)
        away_odds = round(max(1.3, 2.6 + gap * 0.4), 2)

        rows.append({
            "Date": current_date.strftime("%d/%m/%Y"),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": ftr,
            "B365H": home_odds,
            "B365D": draw_odds,
            "B365A": away_odds,
        })
        current_date += timedelta(days=matchday_gap_days)

    return rows


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    all_rows = []
    for year in [2021, 2022, 2023, 2024]:
        all_rows.extend(simulate_season(year))

    fieldnames = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    draws = sum(1 for r in all_rows if r["FTR"] == "D")
    print(f"wrote {len(all_rows)} matches to {OUTPUT_PATH} ({draws} draws, {draws/len(all_rows):.1%})")


if __name__ == "__main__":
    main()
