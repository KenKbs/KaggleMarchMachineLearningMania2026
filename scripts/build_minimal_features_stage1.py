import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

RAW_DIR = Path('data/raw')
CLEAN_DIR = Path('data/cleaned')


def parse_seed_value(seed: str) -> int | float:
    """Convert seed strings like "W01" into an integer [1,16]

    Args:
        seed (str): Raw seed string

    Returns:
        int, float: either int if seed exists, else it's a np.nan (float)
    """
    m = re.search(r"(\d+)", str(seed))
    return int(m.group(1)) if m else np.nan


def assert_seed_parser_examples() -> None:
    """Sanity check if parsing works
    """
    assert parse_seed_value('W01') == 1 
    assert parse_seed_value('X16a') == 16
    assert parse_seed_value('Z12b') == 12


def build_team_game_rows(reg_compact: pd.DataFrame) -> pd.DataFrame:
    """Turn each game row into two rows, one per team, --> aggregate team stats easily later

    Args:
        reg_compact (pd.DataFrame): usually regularSeasonCompactResults table

    Returns:
        pd.DataFrame: w_rows = winner perspective, l_rows = looser perspecctive --> every game = 2 team rows
    """
    w_rows = reg_compact[[
        'Season',
        'DayNum',
        'WTeamID',
        'LTeamID',
        'WScore',
        'LScore',
    ]].copy()
    w_rows = w_rows.rename(
        columns={
            'WTeamID': 'TeamID',
            'LTeamID': 'OppTeamID',
            'WScore': 'TeamScore',
            'LScore': 'OppScore',
        }
    )
    w_rows['Win'] = 1

    l_rows = reg_compact[[
        'Season',
        'DayNum',
        'LTeamID',
        'WTeamID',
        'LScore',
        'WScore',
    ]].copy()
    l_rows = l_rows.rename(
        columns={
            'LTeamID': 'TeamID',
            'WTeamID': 'OppTeamID',
            'LScore': 'TeamScore',
            'WScore': 'OppScore',
        }
    )
    l_rows['Win'] = 0

    rows = pd.concat([w_rows, l_rows], ignore_index=True)
    rows['PointMargin'] = rows['TeamScore'] - rows['OppScore']
    return rows


def build_team_season_features(reg_compact: pd.DataFrame) -> pd.DataFrame:
    """Build a per-team, per-season summary table from regular season games
    Baseline features / metadata per team

    Args:
        reg_compact (pd.DataFrame):   usually regularSeasonCompactResults table

    Returns:
        pd.DataFrame: One row per season, TeamID with important stats per team. 
    """
    team_games = build_team_game_rows(reg_compact)

    season_agg = (
        team_games.groupby(['Season', 'TeamID'], as_index=False)
        .agg(
            Games=('Win', 'size'),
            Wins=('Win', 'sum'),
            TotalPointMargin=('PointMargin', 'sum'),
        )
    )
    season_agg['RegularSeason_WinPct'] = season_agg['Wins'] / season_agg['Games']
    season_agg['RegularSeason_PointMarginPerGame'] = (
        season_agg['TotalPointMargin'] / season_agg['Games']
    )

    team_games = team_games.sort_values(['Season', 'TeamID', 'DayNum'])
    last10 = (
        team_games.groupby(['Season', 'TeamID'], as_index=False)
        .tail(10)
        .groupby(['Season', 'TeamID'], as_index=False)
        .agg(
            Last10Games=('Win', 'size'),
            Last10Wins=('Win', 'sum'),
            Last10PointMargin=('PointMargin', 'mean'),
        )
    )
    last10['Last10_WinPct'] = last10['Last10Wins'] / last10['Last10Games']

    feats = season_agg.merge(last10[['Season', 'TeamID', 'Last10_WinPct', 'Last10PointMargin']], on=['Season', 'TeamID'], how='left')
    feats = feats[[
        'Season',
        'TeamID',
        'RegularSeason_WinPct',
        'RegularSeason_PointMarginPerGame',
        'Last10_WinPct',
        'Last10PointMargin',
    ]].copy()
    return feats


def build_seed_table(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Make a simple (Season, TeamID) --> SeedNum lookup Table

    Args:
        seed_df (pd.DataFrame): usually dataframe created by converting MNCAATTourney.csv seeds

    Returns:
        pd.DataFrame: Seed lookup table with columns (Season, TeamID, SeedNum)
    """
    seeds = seed_df[['Season', 'TeamID', 'Seed']].copy()
    seeds['SeedNum'] = seeds['Seed'].map(parse_seed_value)
    return seeds[['Season', 'TeamID', 'SeedNum']]


def make_canonical_tourney_matchups(tourney_compact: pd.DataFrame) -> pd.DataFrame:
    """Convert hiistorical tournament results into a consistent training dataset format

    Args:
        tourney_compact (pd.DataFrame): usualyl dataframe generated from MNCAATourneyCompactResults.csv

    Returns:
        pd.DataFrame: Team1ID, Team2ID, target (= 1 if Team1 won, 0 if Team 2 won)
    """
    out = tourney_compact[['Season', 'WTeamID', 'LTeamID']].copy()
    out['Team1ID'] = out[['WTeamID', 'LTeamID']].min(axis=1)
    out['Team2ID'] = out[['WTeamID', 'LTeamID']].max(axis=1)
    out['target'] = (out['WTeamID'] == out['Team1ID']).astype(int)
    return out[['Season', 'Team1ID', 'Team2ID', 'target']]


def parse_submission_ids(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Turn submission IDs like 2025_1101_1325 into columns

    Args:
        sub_df (pd.DataFrame): SampleSubmissionStage1.csv as dataframe with an ID column

    Returns:
        pd.DataFrame: Dataframe with raw UD and IDs split into Season, Team1ID, and Team2ID columns 
    """
    ids = sub_df['ID'].str.split('_', expand=True)
    out = pd.DataFrame(
        {
            'ID': sub_df['ID'].values, # Keep raw ID values for submission later
            'Season': ids[0].astype(int),
            'Team1ID': ids[1].astype(int),
            'Team2ID': ids[2].astype(int),
        }
    )
    return out


def attach_pair_features(
    pair_df: pd.DataFrame,
    team_feats: pd.DataFrame,
    seeds: pd.DataFrame,
    include_target: bool,
) -> pd.DataFrame:
    """Builds a feeature dataframe for training

    Args:
        pair_df (pd.DataFrame): rows like: Season, Team1ID, Teams2ID (+ target (who won) if training)
        team_feats (pd.DataFrame): per team season stats
        seeds (pd.DataFrame): (Season, Teamid)--> SeedNum
        include_target (bool): if targets (who won) should be included for training

    Returns:
        pd.DataFrame: Creates a clean feature table SeedDiiff, seed_missing_any, WinPctDiff, PoinntMarginDiff,
                      Team1_Last10_WinPct, Team1_Last10_PointMargin, Team2_Last10_WinPct, Team2_Last10_PointMargin etc. 
    """
    t1 = team_feats.rename(
        columns={
            'TeamID': 'Team1ID',
            'RegularSeason_WinPct': 'Team1_WinPct',
            'RegularSeason_PointMarginPerGame': 'Team1_PointMarginPerGame',
            'Last10_WinPct': 'Team1_Last10_WinPct',
            'Last10PointMargin': 'Team1_Last10_PointMargin',
        }
    )
    t2 = team_feats.rename(
        columns={
            'TeamID': 'Team2ID',
            'RegularSeason_WinPct': 'Team2_WinPct',
            'RegularSeason_PointMarginPerGame': 'Team2_PointMarginPerGame',
            'Last10_WinPct': 'Team2_Last10_WinPct',
            'Last10PointMargin': 'Team2_Last10_PointMargin',
        }
    )

    s1 = seeds.rename(columns={'TeamID': 'Team1ID', 'SeedNum': 'Team1_SeedNum'})
    s2 = seeds.rename(columns={'TeamID': 'Team2ID', 'SeedNum': 'Team2_SeedNum'})

    merged = pair_df.merge(t1, on=['Season', 'Team1ID'], how='left')
    merged = merged.merge(t2, on=['Season', 'Team2ID'], how='left')
    merged = merged.merge(s1, on=['Season', 'Team1ID'], how='left')
    merged = merged.merge(s2, on=['Season', 'Team2ID'], how='left')

    merged['SeedDiff'] = merged['Team1_SeedNum'] - merged['Team2_SeedNum']
    merged['seed_missing_any'] = (
        merged['Team1_SeedNum'].isna() | merged['Team2_SeedNum'].isna()
    ).astype(int)
    merged['WinPctDiff'] = merged['Team1_WinPct'] - merged['Team2_WinPct']
    merged['PointMarginDiff'] = (
        merged['Team1_PointMarginPerGame'] - merged['Team2_PointMarginPerGame']
    )
    merged['Last10WinPctDiff'] = (
        merged['Team1_Last10_WinPct'] - merged['Team2_Last10_WinPct']
    )
    merged['Last10PointMarginDiff'] = (
        merged['Team1_Last10_PointMargin'] - merged['Team2_Last10_PointMargin']
    )

    cols = [
        'Season',
        'Team1ID',
        'Team2ID',
        'SeedDiff',
        'seed_missing_any',
        'WinPctDiff',
        'PointMarginDiff',
        'Last10WinPctDiff',
        'Last10PointMarginDiff',
    ]
    if include_target:
        cols.append('target')

    return merged[cols].copy()


def validate_pair_table(df: pd.DataFrame, with_target: bool, table_name: str) -> None:
    """Quick sanity check to catch errors

    Args:
        df (pd.DataFrame): _description_
        with_target (bool): _description_
        table_name (str): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    key_dupes = df.duplicated(subset=['Season', 'Team1ID', 'Team2ID']).sum()
    if key_dupes:
        raise ValueError(f'{table_name}: duplicate matchup keys found: {key_dupes}')

    if with_target and not set(df['target'].unique()).issubset({0, 1}):
        raise ValueError(f'{table_name}: target contains values outside 0/1')


GenderPrefix = Literal["M", "W"] # Gender prefix for function definition
def build_gender_pipeline(prefix: GenderPrefix) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pipeline to run everything above for men or women
    - loads the three raw CSV's for the three stages of data: RegularSeasonCompactResults, NCAATourneyCompactResults and NCAATourneySeeds.
    - Builds team features + seed table
    - Builds cacnonical tournament matchcups
    - Builds training features + validates them with a quick sanity check
    - team_feats is useful as a lookup table 
    - seed_table useful as a lookup table (season, teamid, seednum)
    - train_feat is the model ready training dataset (X+y)
    
    Args:
        prefix (str): M for men W for women

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: team features, seed table, and train features
    """
    reg = pd.read_csv(RAW_DIR / f'{prefix}RegularSeasonCompactResults.csv')
    tour = pd.read_csv(RAW_DIR / f'{prefix}NCAATourneyCompactResults.csv')
    seeds = pd.read_csv(RAW_DIR / f'{prefix}NCAATourneySeeds.csv')

    team_feats = build_team_season_features(reg)
    seed_table = build_seed_table(seeds)

    train_pairs = make_canonical_tourney_matchups(tour)
    train_feat = attach_pair_features(train_pairs, team_feats, seed_table, include_target=True)
    validate_pair_table(train_feat, with_target=True, table_name=f'{prefix} train')

    return team_feats, seed_table, train_feat


def build_inference_features(
    sample_sub: pd.DataFrame,
    men_team_feats: pd.DataFrame,
    men_seed_table: pd.DataFrame,
    women_team_feats: pd.DataFrame,
    women_seed_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build the feature table for the matchups the competition wants us to predict (from sample submission.csv)

    Args:
        sample_sub (pd.DataFrame): 
        men_team_feats (pd.DataFrame): _description_
        men_seed_table (pd.DataFrame): _description_
        women_team_feats (pd.DataFrame): _description_
        women_seed_table (pd.DataFrame): _description_

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    base = parse_submission_ids(sample_sub)

    men_mask = base['Team1ID'] < 3000
    women_mask = ~men_mask

    men_pairs = base.loc[men_mask].copy()
    women_pairs = base.loc[women_mask].copy()

    men_feat = attach_pair_features(men_pairs, men_team_feats, men_seed_table, include_target=False)
    women_feat = attach_pair_features(women_pairs, women_team_feats, women_seed_table, include_target=False)

    out = pd.concat([men_feat, women_feat], ignore_index=True)
    out = out.sort_values(['Season', 'Team1ID', 'Team2ID']).reset_index(drop=True)
    validate_pair_table(out, with_target=False, table_name='stage1 inference')

    if len(out) != len(sample_sub):
        raise ValueError('stage1 inference: row count mismatch against sample submission')

    return out


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    assert_seed_parser_examples()

    men_team_feats, men_seed_table, men_train = build_gender_pipeline('M')
    women_team_feats, women_seed_table, women_train = build_gender_pipeline('W')

    sample_sub = pd.read_csv(RAW_DIR / 'SampleSubmissionStage1.csv')
    inference = build_inference_features(
        sample_sub,
        men_team_feats,
        men_seed_table,
        women_team_feats,
        women_seed_table,
    )

    men_team_feats.to_csv(CLEAN_DIR / 'men_team_season_features_minimal.csv', index=False)
    women_team_feats.to_csv(CLEAN_DIR / 'women_team_season_features_minimal.csv', index=False)
    men_train.to_csv(CLEAN_DIR / 'men_tourney_train_features_minimal.csv', index=False)
    women_train.to_csv(CLEAN_DIR / 'women_tourney_train_features_minimal.csv', index=False)
    inference.to_csv(CLEAN_DIR / 'stage1_inference_features_minimal.csv', index=False)

    print('Generated:')
    print(' - data/cleaned/men_team_season_features_minimal.csv', len(men_team_feats))
    print(' - data/cleaned/women_team_season_features_minimal.csv', len(women_team_feats))
    print(' - data/cleaned/men_tourney_train_features_minimal.csv', len(men_train))
    print(' - data/cleaned/women_tourney_train_features_minimal.csv', len(women_train))
    print(' - data/cleaned/stage1_inference_features_minimal.csv', len(inference))


if __name__ == '__main__':
    main()
