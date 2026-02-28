import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "SeedDiff",
    "seed_missing_any",
    "WinPctDiff",
    "PointMarginDiff",
    "Last10WinPctDiff",
    "Last10PointMarginDiff",
]
TRAIN_REQUIRED_COLS = ["Season", "Team1ID", "Team2ID", "target", *FEATURE_COLS]
INFER_REQUIRED_COLS = ["Season", "Team1ID", "Team2ID", *FEATURE_COLS]
YEARS_DEFAULT = [2022, 2023, 2024, 2025]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training/evaluation/inference.

    Returns:
        argparse.Namespace: Parsed CLI options including input/output paths,
        validation years, random seed, and number of random-search trials.
    """
    parser = argparse.ArgumentParser(
        description="Rolling Logistic Regression baseline for Stage1 (men/women split)."
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
        default=Path("data/submissions/logreg_stage1_oof_2022_2025.csv"),
    )
    parser.add_argument(
        "--output-preds-path",
        type=Path,
        default=Path("data/submissions/logreg_stage1_predictions.csv"),
    )
    parser.add_argument(
        "--output-metrics-path",
        type=Path,
        default=Path("data/submissions/logreg_stage1_metrics.json"),
    )
    parser.add_argument("--years", type=int, nargs="+", default=YEARS_DEFAULT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=30)
    return parser.parse_args()


def suppress_known_warnings() -> None:
    """Suppress known sklearn logistic-regression deprecation warnings.

    These warnings are currently noisy and do not change runtime behavior for
    this baseline pipeline.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*'penalty' was deprecated.*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Inconsistent values: penalty=l1 with l1_ratio=0.0.*",
        category=UserWarning,
        module=r"sklearn\.linear_model\._logistic",
    )


def assert_required_columns(df: pd.DataFrame, required_cols: list[str], name: str) -> None:
    """Validate that all required columns are present in a DataFrame.

    Args:
        df: DataFrame to validate.
        required_cols: Column names that must exist in `df`.
        name: Human-readable dataset name used in error messages.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def validate_keys(df: pd.DataFrame, name: str) -> None:
    """Validate uniqueness of matchup keys within a dataset.

    Args:
        df: DataFrame expected to contain one row per matchup key.
        name: Human-readable dataset name used in error messages.

    Raises:
        ValueError: If duplicate `(Season, Team1ID, Team2ID)` rows are found.
    """
    dupes = df.duplicated(subset=["Season", "Team1ID", "Team2ID"]).sum()
    if dupes:
        raise ValueError(f"{name}: duplicate (Season, Team1ID, Team2ID) rows: {dupes}")


def validate_target(df: pd.DataFrame, name: str) -> None:
    """Validate that `target` contains only binary labels {0,1}.

    Args:
        df: DataFrame that must include a `target` column.
        name: Human-readable dataset name used in error messages.

    Raises:
        ValueError: If `target` includes values other than 0 or 1.
    """
    unique_vals = set(df["target"].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"{name}: target has values outside {{0,1}}: {sorted(unique_vals)}")


def make_pipeline(params: dict[str, Any], random_state: int) -> Pipeline:
    """Create the sklearn model pipeline used across folds and inference.

    Pipeline steps:
    - `SimpleImputer(strategy="median")` to fill missing feature values.
    - `StandardScaler()` to normalize feature scales.
    - `LogisticRegression(...)` with sampled hyperparameters.

    Args:
        params: Hyperparameter dictionary with keys `C`, `penalty`, `solver`,
            and `class_weight`.
        random_state: Random seed passed into `LogisticRegression`.

    Returns:
        Pipeline: Configured sklearn pipeline instance.
    """
    model = LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        solver=params["solver"],
        class_weight=params["class_weight"],
        max_iter=2000,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def sample_hyperparams(n_iter: int, random_state: int) -> list[dict[str, Any]]:
    """Sample random hyperparameter candidates for logistic regression.

    Current search space is restricted to:
    - `solver="liblinear"`
    - `penalty` in `{"l1", "l2"}`
    - `C` sampled log-uniformly from `[1e-3, 1e2]`
    - `class_weight=None`

    Args:
        n_iter: Number of unique candidate parameter sets to sample.
        random_state: Seed for deterministic sampling.

    Raises:
        RuntimeError: If not enough unique candidates can be produced within
            the configured attempt budget.

    Returns:
        list[dict[str, Any]]: List of hyperparameter dictionaries.
    """
    rng = np.random.default_rng(random_state)
    results: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    max_attempts = max(100, n_iter * 20)
    attempts = 0

    while len(results) < n_iter and attempts < max_attempts:
        attempts += 1
        solver = "liblinear"
        penalty = str(rng.choice(["l1", "l2"]))
        c_value = float(10 ** rng.uniform(-3, 2))
        c_value = float(np.round(c_value, 10))
        key = (solver, penalty, c_value)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "solver": solver,
                "penalty": penalty,
                "class_weight": None,
                "C": c_value,
            }
        )

    if len(results) < n_iter:
        raise RuntimeError(
            f"Could not sample enough unique hyperparams. requested={n_iter}, got={len(results)}"
        )
    return results


def rolling_oof_predict(
    train_df: pd.DataFrame,
    years: list[int],
    params: dict[str, Any],
    random_state: int,
    division_name: str,
) -> pd.DataFrame:
    """Generate rolling out-of-fold predictions for one division.

    For each validation season, the model is trained on seasons `<= year-1`
    and predicts probabilities on season `== year`.

    Args:
        train_df: Division-specific training DataFrame with features and `target`.
        years: Validation years to evaluate.
        params: Hyperparameters passed to `make_pipeline`.
        random_state: Random seed used in model construction.
        division_name: Division label used in error messages and output.

    Raises:
        ValueError: If a fold has empty train/validation rows, fold-year
            integrity fails, validation season content is invalid, or produced
            probabilities are missing/outside `[0,1]`.

    Returns:
        pd.DataFrame: OOF predictions with columns `Division`, `Season`,
        `Team1ID`, `Team2ID`, `target`, `PredProb`, and `PredHard`.
    """
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

        model = make_pipeline(params=params, random_state=random_state)
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


def compute_brier_by_year(oof_df: pd.DataFrame, years: list[int]) -> dict[str, float]:
    """Compute per-year Brier scores from an OOF prediction table.

    Args:
        oof_df: OOF prediction DataFrame containing `target` and `PredProb`.
        years: Seasons to evaluate.

    Raises:
        ValueError: If OOF rows are missing for any requested year.

    Returns:
        dict[str, float]: Mapping `{year_string: brier_score}`.
    """
    metrics: dict[str, float] = {}
    for year in years:
        year_df = oof_df[oof_df["Season"] == year]
        if year_df.empty:
            raise ValueError(f"Missing OOF rows for year={year}")
        metrics[str(year)] = float(brier_score_loss(year_df["target"], year_df["PredProb"]))
    return metrics


def tune_division(
    train_df: pd.DataFrame,
    years: list[int],
    n_iter: int,
    random_state: int,
    division_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Tune hyperparameters for one division via random search.

    This function samples `n_iter` hyperparameter candidates, evaluates each
    candidate with rolling out-of-fold predictions over the provided `years`,
    computes pooled Brier score across all OOF rows, and keeps the best
    (lowest-score) candidate.

    Args:
        train_df: Division-specific training table containing `Season`,
            matchup keys, feature columns used by `rolling_oof_predict`, and
            binary `target` (0/1).
        years: Validation seasons for rolling evaluation (for example
            `[2022, 2023, 2024, 2025]`).
        n_iter: Number of random hyperparameter configurations to evaluate.
        random_state: Seed used for deterministic candidate sampling and model
            reproducibility.
        division_name: Label for logging/error context (for example `"men"` or
            `"women"`).

    Raises:
        RuntimeError: If no valid best candidate is produced (unexpected).

    Returns:
        A tuple `(best_params, best_oof, metrics)` where:
        - `best_params` is the selected hyperparameter dictionary.
        - `best_oof` is the OOF (out of fold) prediction DataFrame for the best candidate
          with columns: `Division`, `Season`, `Team1ID`, `Team2ID`, `target`,
          `PredProb`, `PredHard`.
        - `metrics` is a dictionary with:
          `brier_pooled`, `brier_by_year`, `best_params`, and `n_trials`.
    """
    candidates = sample_hyperparams(n_iter=n_iter, random_state=random_state)
    best_params: dict[str, Any] | None = None
    best_score = float("inf")
    best_oof: pd.DataFrame | None = None

    for params in candidates:
        oof = rolling_oof_predict(
            train_df=train_df,
            years=years,
            params=params,
            random_state=random_state,
            division_name=division_name,
        )
        score = float(brier_score_loss(oof["target"], oof["PredProb"]))
        if score < best_score:
            best_score = score
            best_params = params
            best_oof = oof

    if best_params is None or best_oof is None:
        raise RuntimeError(f"{division_name}: tuning failed to produce a best model")

    metrics = {
        "brier_pooled": best_score,
        "brier_by_year": compute_brier_by_year(best_oof, years),
        "best_params": best_params,
        "n_trials": n_iter,
    }
    return best_params, best_oof, metrics


def predict_stage1_by_year(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    years: list[int],
    params: dict[str, Any],
    random_state: int,
    division_name: str,
) -> pd.DataFrame:
    """Predict Stage1 inference rows year-by-year for one division.

    For each inference season, fit on training seasons `<= year-1` and predict
    probabilities for that season's inference rows.

    Args:
        train_df: Division-specific labeled training DataFrame.
        infer_df: Division-specific unlabeled inference DataFrame.
        years: Seasons to score.
        params: Selected hyperparameters for this division.
        random_state: Random seed used in model construction.
        division_name: Division label used in error messages and output.

    Raises:
        ValueError: If a season has no train data, fold integrity fails, no
            inference rows are available for requested years, or predictions
            contain NaNs.

    Returns:
        pd.DataFrame: Prediction table with columns `Season`, `Team1ID`,
        `Team2ID`, `PredProb`, `PredHard`, and `Division`.
    """
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
        model = make_pipeline(params=params, random_state=random_state)
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
    return pred_df


def add_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add Kaggle submission `ID` column (`Season_Team1ID_Team2ID`).
    # TODO Rework later - currently parse_submission_ids from build_minimal_features get's the raw id, but 
    # attach_pair_features drops that ID and now we have to rebuild it. Maybe rewrite build_minimal_features to keep the ID column instead of parsing and dropping it?

    Args:
        df: DataFrame containing `Season`, `Team1ID`, and `Team2ID`.

    Returns:
        pd.DataFrame: Copy of input DataFrame with an added `ID` column.
    """
    out = df.copy()
    out["ID"] = (
        out["Season"].astype(int).astype(str)
        + "_"
        + out["Team1ID"].astype(int).astype(str)
        + "_"
        + out["Team2ID"].astype(int).astype(str)
    )
    return out


def main() -> None:
    """Run the full baseline pipeline end-to-end.

    Workflow:
    1. Parse CLI args and load input CSVs.
    2. Validate schemas, keys, targets, and requested years.
    3. Split men/women inference rows by TeamID range.
    4. Tune and evaluate each division independently.
    5. Generate Stage1 predictions and write OOF/prediction/metrics outputs.
    """
    args = parse_args()
    suppress_known_warnings()
    years = sorted(set(args.years))
    if not years:
        raise ValueError("years list is empty")
    if args.n_iter <= 0:
        raise ValueError("--n-iter must be > 0")

    men_train = pd.read_csv(args.men_train_path) # default=Path("data/cleaned/men_tourney_train_features_minimal.csv"),
    women_train = pd.read_csv(args.women_train_path) # default=Path("data/cleaned/women_tourney_train_features_minimal.csv") --> The historical data we train on
    stage1 = pd.read_csv(args.stage1_features_path) # default=Path("data/cleaned/stage1_inference_features_minimal.csv")  --> The stuff we want to predict on

    # Validate schemas
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

    # Split inference rows by division using TeamID ranges (men / women)
    men_infer = stage1[stage1["Team1ID"] < 3000].copy()
    women_infer = stage1[stage1["Team1ID"] >= 3000].copy()
    if men_infer.empty or women_infer.empty:
        raise ValueError("stage1_features must include both men and women rows")

    # Find best values for each division independently, then generate predictions for Stage1 inference rows
    men_best_params, men_oof, men_metrics = tune_division(
        train_df=men_train,
        years=years,
        n_iter=args.n_iter,
        random_state=args.random_state,
        division_name="men",
    )
    women_best_params, women_oof, women_metrics = tune_division(
        train_df=women_train,
        years=years,
        n_iter=args.n_iter,
        random_state=args.random_state + 1,
        division_name="women",
    )

    # Generate predictions for Stage1 inference rows using the best hyperparameters found for each division
    men_pred = predict_stage1_by_year(
        train_df=men_train,
        infer_df=men_infer,
        years=years,
        params=men_best_params,
        random_state=args.random_state,
        division_name="men",
    )
    women_pred = predict_stage1_by_year(
        train_df=women_train,
        infer_df=women_infer,
        years=years,
        params=women_best_params,
        random_state=args.random_state + 1,
        division_name="women",
    )

    # Combine
    oof_all = pd.concat([men_oof, women_oof], ignore_index=True).sort_values(
        ["Season", "Team1ID", "Team2ID"]
    )

    # Reconstruct Kaggle submission ID and select output columns for predictions
    preds_all = pd.concat([men_pred, women_pred], ignore_index=True).sort_values(
        ["Season", "Team1ID", "Team2ID"]
    )
    preds_all = add_id_column(preds_all)
    preds_out = preds_all[["ID", "PredProb", "PredHard"]].copy()

    # Validate final output row counts before writing
    expected_infer_rows = len(stage1)
    if len(preds_out) != expected_infer_rows:
        raise ValueError(
            f"Final predictions row mismatch. got={len(preds_out)} expected={expected_infer_rows}"
        )

    # Write outputs
    args.output_oof_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_preds_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    oof_all.to_csv(args.output_oof_path, index=False)
    preds_out.to_csv(args.output_preds_path, index=False)

    metrics = {
        "men": men_metrics,
        "women": women_metrics,
        "years": years,
        "feature_columns": FEATURE_COLS,
    }
    args.output_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Log summary
    print("Generated:")
    print(f" - {args.output_oof_path} rows={len(oof_all)}")
    print(f" - {args.output_preds_path} rows={len(preds_out)}")
    print(f" - {args.output_metrics_path}")
    print(f"men_brier_pooled_2022_2025={men_metrics['brier_pooled']:.6f}")
    print(f"women_brier_pooled_2022_2025={women_metrics['brier_pooled']:.6f}")


if __name__ == "__main__":
    main()
