import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify H2H features for one matchup against raw Kaggle files."
    )
    parser.add_argument("--division", choices=["M", "W"], required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--team1", type=int, required=True)
    parser.add_argument("--team2", type=int, required=True)
    parser.add_argument("--window", type=int, default=3)
    return parser.parse_args()


def canonicalize_games(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out["Source"] = source
    out["Team1ID"] = out[["WTeamID", "LTeamID"]].min(axis=1).astype(int)
    out["Team2ID"] = out[["WTeamID", "LTeamID"]].max(axis=1).astype(int)
    out["Team1Won"] = (out["WTeamID"] == out["Team1ID"]).astype(int)
    out["Team1Margin"] = out["WScore"] - out["LScore"]
    out.loc[out["Team1Won"] == 0, "Team1Margin"] = (
        out.loc[out["Team1Won"] == 0, "LScore"] - out.loc[out["Team1Won"] == 0, "WScore"]
    )
    return out


def main() -> None:
    args = parse_args()
    team1 = min(args.team1, args.team2)
    team2 = max(args.team1, args.team2)

    reg = pd.read_csv(
        RAW_DIR / f"{args.division}RegularSeasonCompactResults.csv",
        usecols=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"],
    )
    tour = pd.read_csv(
        RAW_DIR / f"{args.division}NCAATourneyCompactResults.csv",
        usecols=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"],
    )

    reg = canonicalize_games(reg, "Regular")
    tour = canonicalize_games(tour, "NCAA")

    hist_start = args.season - args.window + 1
    reg_hist = reg[(reg["Season"] >= hist_start) & (reg["Season"] <= args.season)]
    tour_hist = tour[(tour["Season"] >= hist_start) & (tour["Season"] <= args.season - 1)]
    hist = pd.concat([reg_hist, tour_hist], ignore_index=True)

    matchup = hist[(hist["Team1ID"] == team1) & (hist["Team2ID"] == team2)].copy()
    matchup = matchup.sort_values(["Season", "DayNum", "Source"]).reset_index(drop=True)

    t1_wins = int(matchup["Team1Won"].sum())
    t2_wins = int((1 - matchup["Team1Won"]).sum())
    t1_avg = float(matchup.loc[matchup["Team1Won"] == 1, "Team1Margin"].mean()) if t1_wins else 0.0
    t2_avg = float((-matchup.loc[matchup["Team1Won"] == 0, "Team1Margin"]).mean()) if t2_wins else 0.0

    print(f"Matchup: season={args.season}, team1={team1}, team2={team2}, division={args.division}")
    print(f"Window: {args.window} seasons")
    print(f"Included regular seasons: {hist_start}..{args.season}")
    print(f"Included NCAA seasons:    {hist_start}..{args.season - 1}")
    print("")
    if matchup.empty:
        print("No historical games found in window.")
    else:
        print(
            matchup[
                [
                    "Source",
                    "Season",
                    "DayNum",
                    "WTeamID",
                    "LTeamID",
                    "WScore",
                    "LScore",
                    "Team1Won",
                    "Team1Margin",
                ]
            ].to_string(index=False)
        )
    print("")
    print(f"H2H_Team1_Wins_3y: {t1_wins}")
    print(f"H2H_Team2_Wins_3y: {t2_wins}")
    print(f"H2H_Team1_AvgWinMargin_3y: {t1_avg}")
    print(f"H2H_Team2_AvgWinMargin_3y: {t2_avg}")

    cleaned_path = CLEAN_DIR / (
        "men_tourney_train_features_minimal.csv"
        if args.division == "M"
        else "women_tourney_train_features_minimal.csv"
    )
    cleaned = pd.read_csv(cleaned_path)
    row = cleaned[
        (cleaned["Season"] == args.season)
        & (cleaned["Team1ID"] == team1)
        & (cleaned["Team2ID"] == team2)
    ]
    print("")
    if row.empty:
        print("No matching row found in cleaned tournament training feature file.")
    else:
        print("Row from cleaned training features:")
        print(
            row[
                [
                    "Season",
                    "Team1ID",
                    "Team2ID",
                    "H2H_Team1_Wins_3y",
                    "H2H_Team2_Wins_3y",
                    "H2H_Team1_AvgWinMargin_3y",
                    "H2H_Team2_AvgWinMargin_3y",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
