"""
Turns the raw numeric features for a fixture into plain-language analysis
bullets ("why the model likes this as a draw") and a confidence label.

This is rule-based on top of the model's own features - it doesn't change
what the model predicts, it just explains the prediction so a pick isn't
just a bare percentage. Each rule mirrors something genuinely linked to
draws in football analytics (tight standings, low combined goal output,
draw-heavy head-to-head history, etc.).
"""

# thresholds tuned to flag "notable" situations, not just any non-zero value
PPG_GAP_TIGHT = 0.6          # recent form points-per-game gap below this = "evenly matched"
SEASON_POINTS_GAP_TIGHT = 6  # season points gap below this = "close in the table"
LOW_GOAL_EXPECTATION = 2.6   # combined recent goals-for below this = "low-scoring sides"
HIGH_H2H_DRAW_RATE = 0.30    # head-to-head draw rate above this = "history of draws"
SEASON_GD_GAP_TIGHT = 5      # goal difference gap below this = "similar quality"


def build_analysis(features, draw_probability):
    """
    features: dict produced by features.build_fixture_features()
    draw_probability: float, the model's calibrated probability

    Returns: {"confidence": str, "factors": [str, ...]}
    """
    factors = []

    if features.get("ppg_gap", 99) <= PPG_GAP_TIGHT:
        factors.append("Both teams are in similar recent form (close points-per-game over their last games)")

    if features.get("season_points_gap", 99) <= SEASON_POINTS_GAP_TIGHT:
        factors.append("Closely matched in the table - only a small points gap separates them")

    if features.get("season_gd_gap", 99) <= SEASON_GD_GAP_TIGHT:
        factors.append("Similar goal difference this season, suggesting comparable overall quality")

    if features.get("combined_goal_expectation", 99) <= LOW_GOAL_EXPECTATION:
        factors.append("Both sides have been low-scoring recently, raising the odds of a stalemate")

    if features.get("h2h_draw_rate", 0) >= HIGH_H2H_DRAW_RATE:
        rate = features["h2h_draw_rate"]
        factors.append(f"These two have drawn {rate:.0%} of their recent head-to-head meetings")

    bookmaker_prob = features.get("bookmaker_draw_prob")
    if bookmaker_prob is not None:
        if bookmaker_prob >= draw_probability:
            factors.append("The betting market is pricing the draw even more confidently than the model")
        else:
            factors.append("The model sees more draw value here than the betting market does")

    if not factors:
        factors.append("Flagged mainly on the model's combined statistical signal rather than one standout factor")

    if draw_probability >= 0.38:
        confidence = "High"
    elif draw_probability >= 0.30:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {"confidence": confidence, "factors": factors}
