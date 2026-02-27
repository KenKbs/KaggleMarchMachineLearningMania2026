from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "cleaned"
SUB_DIR = ROOT / "data" / "submissions"

REQ_REG_COLS = {"Season", "WTeamID", "WScore", "LTeamID", "LScore"}
REQ_TOUR_COLS = {"Season", "WTeamID", "LTeamID"}
REQ_SEED_COLS = {"Season", "Seed", "TeamID"}
REQ_SUB_COLS = {"ID", "Pred"}

FEATURE_DIFF_COLS = [
    "diff_rs_win_pct",
    "diff_rs_score_diff_avg",
    "diff_rs_points_for_avg",
    "diff_rs_points_against_avg",
    "diff_seed_num",
    "diff_has_seed",
]


def assert_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def parse_seed(seed: str) -> float:
    m = re.search(r"(\d+)", str(seed))
    return float(m.group(1)) if m else np.nan


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reg = pd.read_csv(RAW_DIR / "MRegularSeasonCompactResults.csv")
    tour = pd.read_csv(RAW_DIR / "MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv(RAW_DIR / "MNCAATourneySeeds.csv")
    sub = pd.read_csv(RAW_DIR / "SampleSubmissionStage1.csv")

    assert_columns(reg, REQ_REG_COLS, "MRegularSeasonCompactResults.csv")
    assert_columns(tour, REQ_TOUR_COLS, "MNCAATourneyCompactResults.csv")
    assert_columns(seeds, REQ_SEED_COLS, "MNCAATourneySeeds.csv")
    assert_columns(sub, REQ_SUB_COLS, "SampleSubmissionStage1.csv")

    return reg, tour, seeds, sub


def build_team_season_features(reg: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    winners = pd.DataFrame(
        {
            "Season": reg["Season"].astype(int),
            "TeamID": reg["WTeamID"].astype(int),
            "is_win": 1,
            "points_for": reg["WScore"].astype(float),
            "points_against": reg["LScore"].astype(float),
        }
    )
    losers = pd.DataFrame(
        {
            "Season": reg["Season"].astype(int),
            "TeamID": reg["LTeamID"].astype(int),
            "is_win": 0,
            "points_for": reg["LScore"].astype(float),
            "points_against": reg["WScore"].astype(float),
        }
    )
    long_df = pd.concat([winners, losers], ignore_index=True)
    long_df["score_diff"] = long_df["points_for"] - long_df["points_against"]

    grouped = (
        long_df.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            rs_games=("is_win", "size"),
            rs_wins=("is_win", "sum"),
            rs_points_for_avg=("points_for", "mean"),
            rs_points_against_avg=("points_against", "mean"),
            rs_score_diff_avg=("score_diff", "mean"),
        )
        .sort_values(["Season", "TeamID"])
    )
    grouped["rs_losses"] = grouped["rs_games"] - grouped["rs_wins"]
    grouped["rs_win_pct"] = grouped["rs_wins"] / grouped["rs_games"]

    seeds_df = seeds[["Season", "TeamID", "Seed"]].copy()
    seeds_df["Season"] = seeds_df["Season"].astype(int)
    seeds_df["TeamID"] = seeds_df["TeamID"].astype(int)
    seeds_df["seed_num"] = seeds_df["Seed"].map(parse_seed)
    seeds_df["has_seed"] = (~seeds_df["seed_num"].isna()).astype(int)
    seeds_df = seeds_df[["Season", "TeamID", "seed_num", "has_seed"]]

    feat = grouped.merge(seeds_df, on=["Season", "TeamID"], how="left")
    feat["has_seed"] = feat["has_seed"].fillna(0).astype(int)

    if feat.duplicated(["Season", "TeamID"]).any():
        raise ValueError("Duplicate keys found in team-season features")

    cols = [
        "Season",
        "TeamID",
        "rs_games",
        "rs_wins",
        "rs_losses",
        "rs_win_pct",
        "rs_points_for_avg",
        "rs_points_against_avg",
        "rs_score_diff_avg",
        "seed_num",
        "has_seed",
    ]
    return feat[cols]


def parse_submission_ids(sub: pd.DataFrame) -> pd.DataFrame:
    parts = sub["ID"].astype(str).str.extract(r"^(\d{4})_(\d+)_(\d+)$")
    if parts.isna().any().any():
        raise ValueError("SampleSubmissionStage1.csv contains invalid ID format")
    out = sub[["ID"]].copy()
    out["Season"] = parts[0].astype(int)
    out["TeamLowID"] = parts[1].astype(int)
    out["TeamHighID"] = parts[2].astype(int)
    if not (out["TeamLowID"] < out["TeamHighID"]).all():
        raise ValueError("Sample submission IDs are not in canonical low/high order")
    return out


def add_pairwise_features(
    pair_df: pd.DataFrame, team_feat: pd.DataFrame, include_label: bool = True
) -> pd.DataFrame:
    base_cols = [
        "Season",
        "TeamID",
        "rs_win_pct",
        "rs_score_diff_avg",
        "rs_points_for_avg",
        "rs_points_against_avg",
        "seed_num",
        "has_seed",
    ]
    low_feat = team_feat[base_cols].rename(
        columns={
            "TeamID": "TeamLowID",
            "rs_win_pct": "low_rs_win_pct",
            "rs_score_diff_avg": "low_rs_score_diff_avg",
            "rs_points_for_avg": "low_rs_points_for_avg",
            "rs_points_against_avg": "low_rs_points_against_avg",
            "seed_num": "low_seed_num",
            "has_seed": "low_has_seed",
        }
    )
    high_feat = team_feat[base_cols].rename(
        columns={
            "TeamID": "TeamHighID",
            "rs_win_pct": "high_rs_win_pct",
            "rs_score_diff_avg": "high_rs_score_diff_avg",
            "rs_points_for_avg": "high_rs_points_for_avg",
            "rs_points_against_avg": "high_rs_points_against_avg",
            "seed_num": "high_seed_num",
            "has_seed": "high_has_seed",
        }
    )

    merged = pair_df.merge(low_feat, on=["Season", "TeamLowID"], how="left")
    merged = merged.merge(high_feat, on=["Season", "TeamHighID"], how="left")

    merged["diff_rs_win_pct"] = merged["low_rs_win_pct"] - merged["high_rs_win_pct"]
    merged["diff_rs_score_diff_avg"] = merged["low_rs_score_diff_avg"] - merged["high_rs_score_diff_avg"]
    merged["diff_rs_points_for_avg"] = merged["low_rs_points_for_avg"] - merged["high_rs_points_for_avg"]
    merged["diff_rs_points_against_avg"] = (
        merged["low_rs_points_against_avg"] - merged["high_rs_points_against_avg"]
    )
    merged["diff_seed_num"] = merged["low_seed_num"] - merged["high_seed_num"]
    merged["diff_has_seed"] = merged["low_has_seed"] - merged["high_has_seed"]

    out_cols = ["Season", "TeamLowID", "TeamHighID"]
    if include_label:
        out_cols.append("y")
    out_cols.extend(FEATURE_DIFF_COLS)
    return merged[out_cols]


def build_tourney_train_pairs(tour: pd.DataFrame, team_feat: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.DataFrame(
        {
            "Season": tour["Season"].astype(int),
            "WTeamID": tour["WTeamID"].astype(int),
            "LTeamID": tour["LTeamID"].astype(int),
        }
    )
    pairs["TeamLowID"] = pairs[["WTeamID", "LTeamID"]].min(axis=1)
    pairs["TeamHighID"] = pairs[["WTeamID", "LTeamID"]].max(axis=1)
    pairs["y"] = (pairs["WTeamID"] == pairs["TeamLowID"]).astype(int)

    if not (pairs["TeamLowID"] < pairs["TeamHighID"]).all():
        raise ValueError("Training pairs violate TeamLowID < TeamHighID")
    if not pairs["y"].isin([0, 1]).all():
        raise ValueError("Label y contains values outside {0,1}")

    feat_pairs = add_pairwise_features(
        pairs[["Season", "TeamLowID", "TeamHighID", "y"]], team_feat, include_label=True
    )
    return feat_pairs


def rolling_cv_scores(train_df: pd.DataFrame) -> list[dict[str, float | int]]:
    cv_rows: list[dict[str, float | int]] = []
    seasons = sorted(train_df["Season"].unique())
    for val_season in seasons:
        tr = train_df[train_df["Season"] < val_season]
        va = train_df[train_df["Season"] == val_season]
        if tr.empty or va.empty:
            continue

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        C=1.0,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(tr[FEATURE_DIFF_COLS], tr["y"])
        pred = model.predict_proba(va[FEATURE_DIFF_COLS])[:, 1]
        score = log_loss(va["y"], pred, labels=[0, 1])
        cv_rows.append(
            {
                "val_season": int(val_season),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "log_loss": float(score),
            }
        )
    return cv_rows


def train_and_score(train_pairs: pd.DataFrame) -> tuple[Pipeline, dict[str, float | int], list[dict[str, float | int]]]:
    train_pre_2024 = train_pairs[train_pairs["Season"] < 2024].copy()
    holdout_2024 = train_pairs[train_pairs["Season"] == 2024].copy()
    if train_pre_2024.empty or holdout_2024.empty:
        raise ValueError("Need non-empty training data (<2024) and holdout data (2024)")

    cv_rows = rolling_cv_scores(train_pre_2024)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_pre_2024[FEATURE_DIFF_COLS], train_pre_2024["y"])
    holdout_pred = model.predict_proba(holdout_2024[FEATURE_DIFF_COLS])[:, 1]
    holdout_ll = log_loss(holdout_2024["y"], holdout_pred, labels=[0, 1])

    metrics = {
        "train_rows_pre_2024": int(len(train_pre_2024)),
        "holdout_rows_2024": int(len(holdout_2024)),
        "holdout_log_loss_2024": float(holdout_ll),
        "cv_fold_count": int(len(cv_rows)),
        "cv_log_loss_mean": float(np.mean([r["log_loss"] for r in cv_rows])) if cv_rows else np.nan,
    }
    return model, metrics, cv_rows


def build_inference_pairs(sub: pd.DataFrame, team_feat: pd.DataFrame) -> pd.DataFrame:
    base = parse_submission_ids(sub)
    return add_pairwise_features(base, team_feat, include_label=False)


def run() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    reg, tour, seeds, sub = load_inputs()

    team_feat = build_team_season_features(reg, seeds)
    if team_feat["Season"].eq(2024).sum() == 0:
        raise ValueError("No 2024 team-season features found; cannot run holdout")

    train_pairs = build_tourney_train_pairs(tour, team_feat)
    model, metrics, cv_rows = train_and_score(train_pairs)

    inf_pairs = build_inference_pairs(sub, team_feat)
    pred = model.predict_proba(inf_pairs[FEATURE_DIFF_COLS])[:, 1]

    if np.any(pred < 0) or np.any(pred > 1):
        raise ValueError("Predictions outside [0,1]")

    submission = sub[["ID"]].copy()
    submission["Pred"] = pred

    if len(submission) != len(sub):
        raise ValueError("Submission row count mismatch")
    if list(submission.columns) != ["ID", "Pred"]:
        raise ValueError("Submission columns must be exactly ['ID', 'Pred']")

    feature_path = CLEAN_DIR / "men_team_season_features.csv"
    train_path = CLEAN_DIR / "men_tourney_train_pairs.csv"
    infer_path = CLEAN_DIR / "men_stage1_inference_pairs.csv"
    sub_path = SUB_DIR / "baseline_men_logreg_stage1.csv"
    metrics_path = CLEAN_DIR / "men_baseline_metrics.json"
    cv_path = CLEAN_DIR / "men_baseline_cv_scores.csv"

    team_feat.to_csv(feature_path, index=False)
    train_pairs.to_csv(train_path, index=False)
    inf_pairs.to_csv(infer_path, index=False)
    submission.to_csv(sub_path, index=False)
    pd.DataFrame(cv_rows).to_csv(cv_path, index=False)

    metrics["leakage_note"] = "Features built exclusively from MRegularSeasonCompactResults.csv and MNCAATourneySeeds.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"- {feature_path}")
    print(f"- {train_path}")
    print(f"- {infer_path}")
    print(f"- {sub_path}")
    print(f"- {cv_path}")
    print(f"- {metrics_path}")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
