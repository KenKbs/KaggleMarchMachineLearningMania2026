import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

from baseline_logreg_stage1 import (
    FEATURE_COLS,
    INFER_REQUIRED_COLS,
    TRAIN_REQUIRED_COLS,
    YEARS_DEFAULT,
    add_id_column,
    assert_required_columns,
    compute_brier_by_year,
    validate_keys,
    validate_target,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for XGBoost training/evaluation/inference."""
    parser = argparse.ArgumentParser(
        description="Rolling XGBoost baseline for Stage1 (men/women split)."
    )
    parser.add_argument(
        "--men-train-path",
        type=Path,
        default=Path("data/cleaned/men_tourney_train_features_minimal.csv"),
    )
    parser.add_argument(
        "--women-train-path",
        type=Path,
        default=Path("data/cleaned/women_tourney_train_features_minimal.csv"),
    )
    parser.add_argument(
        "--stage1-features-path",
        type=Path,
        default=Path("data/cleaned/stage1_inference_features_minimal.csv"),
    )
    parser.add_argument(
        "--output-oof-path",
        type=Path,
        default=Path("data/submissions/xgb_stage1_oof_2022_2025.csv"),
    )
    parser.add_argument(
        "--output-preds-path",
        type=Path,
        default=Path("data/submissions/xgb_stage1_predictions.csv"),
    )
    parser.add_argument(
        "--output-metrics-path",
        type=Path,
        default=Path("data/submissions/xgb_stage1_metrics.json"),
    )
    parser.add_argument(
        "--output-trials-path",
        type=Path,
        default=Path("data/cleaned/xgb_stage1_trials.csv"),
    )
    parser.add_argument("--years", type=int, nargs="+", default=YEARS_DEFAULT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="0=silent, 1=trial progress, 2=trial + fold/year progress",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print trial progress every N iterations (when --verbose >= 1).",
    )
    parser.add_argument(
        "--xgb-verbosity",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="XGBoost internal verbosity (0=silent, 1=warning, 2=info, 3=debug).",
    )
    return parser.parse_args()


def make_xgb_model(
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
    xgb_verbosity: int,
) -> XGBClassifier:
    """Create the XGBoost model used for fold training and inference."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=xgb_verbosity,
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
    )


def sample_xgb_hyperparams(n_iter: int, random_state: int) -> list[dict[str, Any]]:
    """Sample random hyperparameter candidates for XGBoost."""
    rng = np.random.default_rng(random_state)
    results: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    max_attempts = max(1000, n_iter * 50)
    attempts = 0

    while len(results) < n_iter and attempts < max_attempts:
        attempts += 1
        params: dict[str, Any] = {
            "n_estimators": int(rng.integers(500, 5000)),
            "learning_rate": float(np.round(np.exp(rng.uniform(np.log(0.001), np.log(0.50))), 10)),
            "max_depth": int(rng.integers(1, 30)),
            "min_child_weight": float(
                np.round(np.exp(rng.uniform(np.log(0.5), np.log(20.0))), 10)
            ),
            "subsample": float(np.round(rng.uniform(0.6, 1.0), 10)),
            "colsample_bytree": float(np.round(rng.uniform(0.6, 1.0), 10)),
            "gamma": 0.0 if rng.random() < 0.2 else float(np.round(np.exp(rng.uniform(np.log(1e-8), np.log(5.0))), 10)),
            "reg_alpha": 0.0 if rng.random() < 0.2 else float(np.round(np.exp(rng.uniform(np.log(1e-8), np.log(10.0))), 10)),
            "reg_lambda": 0.0 if rng.random() < 0.2 else float(np.round(np.exp(rng.uniform(np.log(1e-3), np.log(50.0))), 10)),
        }
        key = (
            params["n_estimators"],
            params["learning_rate"],
            params["max_depth"],
            params["min_child_weight"],
            params["subsample"],
            params["colsample_bytree"],
            params["gamma"],
            params["reg_alpha"],
            params["reg_lambda"],
        )
        if key in seen:
            continue
        seen.add(key)
        results.append(params)

    if len(results) < n_iter:
        raise RuntimeError(
            f"Could not sample enough unique hyperparams. requested={n_iter}, got={len(results)}"
        )
    return results


def rolling_oof_predict_xgb(
    train_df: pd.DataFrame,
    years: list[int],
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
    xgb_verbosity: int,
    verbose_level: int,
    division_name: str,
) -> pd.DataFrame:
    """Generate rolling out-of-fold predictions for one division with XGBoost."""
    rows: list[pd.DataFrame] = []
    for year in years:
        train_fold = train_df[train_df["Season"] <= (year - 1)].copy()
        valid_fold = train_df[train_df["Season"] == year].copy()
        if train_fold.empty or valid_fold.empty:
            raise ValueError(
                f"{division_name}: fold year={year} has empty train/valid. "
                f"train_rows={len(train_fold)} valid_rows={len(valid_fold)}"
            )
        train_max_season = int(train_fold["Season"].max())
        if train_max_season > year - 1:
            raise ValueError(
                f"{division_name}: fold integrity failed for year={year}. "
                f"train_max_season={train_max_season}, expected_at_most={year - 1}"
            )
        if set(valid_fold["Season"].unique()) != {year}:
            raise ValueError(f"{division_name}: valid fold has unexpected seasons for year={year}")

        if verbose_level >= 2:
            print(
                f"[{division_name}] year={year}: train_rows={len(train_fold)} "
                f"valid_rows={len(valid_fold)}"
            )

        model = make_xgb_model(
            params=params,
            random_state=random_state,
            n_jobs=n_jobs,
            xgb_verbosity=xgb_verbosity,
        )
        model.fit(train_fold[FEATURE_COLS], train_fold["target"])
        probs = model.predict_proba(valid_fold[FEATURE_COLS])[:, 1]

        fold_out = valid_fold[["Season", "Team1ID", "Team2ID", "target"]].copy()
        fold_out["PredProb"] = probs
        fold_out["PredHard"] = (fold_out["PredProb"] >= 0.5).astype(int)
        fold_out["Division"] = division_name
        rows.append(fold_out)

    out = pd.concat(rows, ignore_index=True)
    if out["PredProb"].isna().any():
        raise ValueError(f"{division_name}: NaN predictions found in OOF")
    if ((out["PredProb"] < 0) | (out["PredProb"] > 1)).any():
        raise ValueError(f"{division_name}: PredProb out of [0,1]")
    return out[
        ["Division", "Season", "Team1ID", "Team2ID", "target", "PredProb", "PredHard"]
    ].copy()


def tune_division_xgb(
    train_df: pd.DataFrame,
    years: list[int],
    n_iter: int,
    random_state: int,
    n_jobs: int,
    xgb_verbosity: int,
    verbose_level: int,
    progress_every: int,
    division_name: str,
    run_id: str,
    run_timestamp_utc: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Tune XGBoost hyperparameters for one division via random search."""
    candidates = sample_xgb_hyperparams(n_iter=n_iter, random_state=random_state)

    best_params: dict[str, Any] | None = None
    best_score = float("inf")
    best_oof: pd.DataFrame | None = None
    best_trial_index: int | None = None
    trial_rows: list[dict[str, Any]] = []
    tune_start = perf_counter()

    if verbose_level >= 1:
        print(
            f"[{division_name}] tuning start: n_iter={n_iter}, n_jobs={n_jobs}, "
            f"xgb_verbosity={xgb_verbosity}"
        )

    years_str = "|".join(str(y) for y in years)
    for i, params in enumerate(candidates, start=1):
        oof = rolling_oof_predict_xgb(
            train_df=train_df,
            years=years,
            params=params,
            random_state=random_state,
            n_jobs=n_jobs,
            xgb_verbosity=xgb_verbosity,
            verbose_level=verbose_level,
            division_name=division_name,
        )
        brier_pooled = float(brier_score_loss(oof["target"], oof["PredProb"]))
        logloss_pooled = float(log_loss(oof["target"], oof["PredProb"]))
        brier_by_year = compute_brier_by_year(oof, years)

        trial_row: dict[str, Any] = {
            "run_id": run_id,
            "run_timestamp_utc": run_timestamp_utc,
            "model": "xgboost",
            "division": division_name,
            "trial_index": i,
            "random_state": random_state,
            "years": years_str,
            "brier_pooled": brier_pooled,
            "logloss_pooled": logloss_pooled,
            "is_best_in_division_run": False,
            **params,
        }
        for year in years:
            trial_row[f"brier_year_{year}"] = float(brier_by_year[str(year)])

        trial_rows.append(trial_row)

        is_new_best = brier_pooled < best_score
        if is_new_best:
            best_score = brier_pooled
            best_params = params
            best_oof = oof
            best_trial_index = i

        if verbose_level >= 1 and (
            i == 1 or i == n_iter or is_new_best or (i % progress_every == 0)
        ):
            elapsed = perf_counter() - tune_start
            print(
                f"[{division_name}] trial {i}/{n_iter} "
                f"brier={brier_pooled:.6f} logloss={logloss_pooled:.6f} "
                f"best_brier={best_score:.6f}"
                f"{' *best' if is_new_best else ''} elapsed_s={elapsed:.1f}"
            )

    if best_params is None or best_oof is None or best_trial_index is None:
        raise RuntimeError(f"{division_name}: tuning failed to produce a best model")

    trial_rows[best_trial_index - 1]["is_best_in_division_run"] = True
    trials_df = pd.DataFrame(trial_rows)
    metrics = {
        "brier_pooled": best_score,
        "brier_by_year": compute_brier_by_year(best_oof, years),
        "best_params": best_params,
        "n_trials": n_iter,
    }
    if verbose_level >= 1:
        print(
            f"[{division_name}] tuning done: best_brier={best_score:.6f}, "
            f"best_trial={best_trial_index}/{n_iter}"
        )
    return best_params, best_oof, metrics, trials_df


def predict_stage1_by_year_xgb(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    years: list[int],
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
    xgb_verbosity: int,
    verbose_level: int,
    division_name: str,
) -> pd.DataFrame:
    """Predict Stage1 rows year-by-year for one division with XGBoost."""
    parts: list[pd.DataFrame] = []
    for year in years:
        infer_year = infer_df[infer_df["Season"] == year].copy()
        if infer_year.empty:
            continue
        train_fold = train_df[train_df["Season"] <= (year - 1)].copy()
        if train_fold.empty:
            raise ValueError(f"{division_name}: no train rows available for inference year={year}")
        train_max_season = int(train_fold["Season"].max())
        if train_max_season > year - 1:
            raise ValueError(
                f"{division_name}: inference fold integrity failed for year={year}. "
                f"train_max_season={train_max_season}, expected_at_most={year - 1}"
            )

        if verbose_level >= 2:
            print(
                f"[{division_name}] inference year={year}: train_rows={len(train_fold)} "
                f"infer_rows={len(infer_year)}"
            )

        model = make_xgb_model(
            params=params,
            random_state=random_state,
            n_jobs=n_jobs,
            xgb_verbosity=xgb_verbosity,
        )
        model.fit(train_fold[FEATURE_COLS], train_fold["target"])
        probs = model.predict_proba(infer_year[FEATURE_COLS])[:, 1]

        out = infer_year[["Season", "Team1ID", "Team2ID"]].copy()
        out["PredProb"] = probs
        out["PredHard"] = (out["PredProb"] >= 0.5).astype(int)
        out["Division"] = division_name
        parts.append(out)

    if not parts:
        raise ValueError(f"{division_name}: no inference rows for requested years={years}")

    pred_df = pd.concat(parts, ignore_index=True)
    if pred_df["PredProb"].isna().any():
        raise ValueError(f"{division_name}: NaN predictions found for Stage1 inference")
    if ((pred_df["PredProb"] < 0) | (pred_df["PredProb"] > 1)).any():
        raise ValueError(f"{division_name}: PredProb out of [0,1] for Stage1 inference")
    return pred_df


def append_trials_history(trials_df: pd.DataFrame, output_path: Path) -> int:
    """Append trial rows to a persistent CSV file and return final row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = pd.read_csv(output_path)
        all_cols = list(dict.fromkeys([*existing.columns, *trials_df.columns]))
        combined = pd.concat(
            [
                existing.reindex(columns=all_cols),
                trials_df.reindex(columns=all_cols),
            ],
            ignore_index=True,
        )
    else:
        combined = trials_df.copy()

    combined.to_csv(output_path, index=False)
    return len(combined)


def main() -> None:
    """Run the full XGBoost baseline pipeline end-to-end."""
    args = parse_args()
    years = sorted(set(args.years))
    if not years:
        raise ValueError("years list is empty")
    if args.n_iter <= 0:
        raise ValueError("--n-iter must be > 0")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs must be != 0")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")

    run_id = uuid.uuid4().hex
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()

    if args.verbose >= 1:
        print(
            f"Run config: years={years} n_iter={args.n_iter} n_jobs={args.n_jobs} "
            f"verbose={args.verbose} progress_every={args.progress_every} "
            f"xgb_verbosity={args.xgb_verbosity}"
        )

    men_train = pd.read_csv(args.men_train_path)
    women_train = pd.read_csv(args.women_train_path)
    stage1 = pd.read_csv(args.stage1_features_path)

    assert_required_columns(men_train, TRAIN_REQUIRED_COLS, "men_train")
    assert_required_columns(women_train, TRAIN_REQUIRED_COLS, "women_train")
    assert_required_columns(stage1, INFER_REQUIRED_COLS, "stage1_features")
    validate_keys(men_train, "men_train")
    validate_keys(women_train, "women_train")
    validate_keys(stage1, "stage1_features")
    validate_target(men_train, "men_train")
    validate_target(women_train, "women_train")

    for y in years:
        if men_train[men_train["Season"] == y].empty:
            raise ValueError(f"men_train: missing validation year={y}")
        if women_train[women_train["Season"] == y].empty:
            raise ValueError(f"women_train: missing validation year={y}")
        if stage1[stage1["Season"] == y].empty:
            raise ValueError(f"stage1_features: missing inference year={y}")

    men_infer = stage1[stage1["Team1ID"] < 3000].copy()
    women_infer = stage1[stage1["Team1ID"] >= 3000].copy()
    if men_infer.empty or women_infer.empty:
        raise ValueError("stage1_features must include both men and women rows")

    men_best_params, men_oof, men_metrics, men_trials = tune_division_xgb(
        train_df=men_train,
        years=years,
        n_iter=args.n_iter,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        xgb_verbosity=args.xgb_verbosity,
        verbose_level=args.verbose,
        progress_every=args.progress_every,
        division_name="men",
        run_id=run_id,
        run_timestamp_utc=run_timestamp_utc,
    )
    women_best_params, women_oof, women_metrics, women_trials = tune_division_xgb(
        train_df=women_train,
        years=years,
        n_iter=args.n_iter,
        random_state=args.random_state + 1,
        n_jobs=args.n_jobs,
        xgb_verbosity=args.xgb_verbosity,
        verbose_level=args.verbose,
        progress_every=args.progress_every,
        division_name="women",
        run_id=run_id,
        run_timestamp_utc=run_timestamp_utc,
    )

    all_trials = pd.concat([men_trials, women_trials], ignore_index=True)
    total_trial_rows = append_trials_history(all_trials, args.output_trials_path)

    men_pred = predict_stage1_by_year_xgb(
        train_df=men_train,
        infer_df=men_infer,
        years=years,
        params=men_best_params,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        xgb_verbosity=args.xgb_verbosity,
        verbose_level=args.verbose,
        division_name="men",
    )
    women_pred = predict_stage1_by_year_xgb(
        train_df=women_train,
        infer_df=women_infer,
        years=years,
        params=women_best_params,
        random_state=args.random_state + 1,
        n_jobs=args.n_jobs,
        xgb_verbosity=args.xgb_verbosity,
        verbose_level=args.verbose,
        division_name="women",
    )

    oof_all = pd.concat([men_oof, women_oof], ignore_index=True).sort_values(
        ["Season", "Team1ID", "Team2ID"]
    )

    preds_all = pd.concat([men_pred, women_pred], ignore_index=True).sort_values(
        ["Season", "Team1ID", "Team2ID"]
    )
    preds_all = add_id_column(preds_all)
    preds_out = preds_all[["ID", "PredProb", "PredHard"]].copy()

    expected_infer_rows = len(stage1)
    if len(preds_out) != expected_infer_rows:
        raise ValueError(
            f"Final predictions row mismatch. got={len(preds_out)} expected={expected_infer_rows}"
        )

    args.output_oof_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_preds_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    oof_all.to_csv(args.output_oof_path, index=False)
    preds_out.to_csv(args.output_preds_path, index=False)

    metrics = {
        "men": men_metrics,
        "women": women_metrics,
        "overall": {
            "brier_pooled": float(brier_score_loss(oof_all["target"], oof_all["PredProb"]))
        },
        "years": years,
        "feature_columns": FEATURE_COLS,
    }
    args.output_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Generated:")
    print(f" - {args.output_oof_path} rows={len(oof_all)}")
    print(f" - {args.output_preds_path} rows={len(preds_out)}")
    print(f" - {args.output_metrics_path}")
    print(f" - {args.output_trials_path} total_rows={total_trial_rows} appended_rows={len(all_trials)}")
    print(f"men_brier_pooled_2022_2025={men_metrics['brier_pooled']:.6f}")
    print(f"women_brier_pooled_2022_2025={women_metrics['brier_pooled']:.6f}")
    print(
        "overall_brier_pooled_2022_2025="
        f"{metrics['overall']['brier_pooled']:.6f}"
    )


if __name__ == "__main__":
    main()
