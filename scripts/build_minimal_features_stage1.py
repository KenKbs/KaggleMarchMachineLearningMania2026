import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path('data/raw')
CLEAN_DIR = Path('data/cleaned')


def parse_seed_value(seed: str):
    m = re.search(r"(\d+)", str(seed))
    return int(m.group(1)) if m else np.nan


def assert_seed_parser_examples() -> None:
    assert parse_seed_value('W01') == 1
    assert parse_seed_value('X16a') == 16
    assert parse_seed_value('Z12b') == 12


def build_team_game_rows(reg_compact: pd.DataFrame) -> pd.DataFrame:
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
    seeds = seed_df[['Season', 'TeamID', 'Seed']].copy()
    seeds['SeedNum'] = seeds['Seed'].map(parse_seed_value)
    return seeds[['Season', 'TeamID', 'SeedNum']]


def make_canonical_tourney_matchups(tourney_compact: pd.DataFrame) -> pd.DataFrame:
    out = tourney_compact[['Season', 'WTeamID', 'LTeamID']].copy()
    out['Team1ID'] = out[['WTeamID', 'LTeamID']].min(axis=1)
    out['Team2ID'] = out[['WTeamID', 'LTeamID']].max(axis=1)
    out['target'] = (out['WTeamID'] == out['Team1ID']).astype(int)
    return out[['Season', 'Team1ID', 'Team2ID', 'target']]


def parse_submission_ids(sub_df: pd.DataFrame) -> pd.DataFrame:
    ids = sub_df['ID'].str.split('_', expand=True)
    out = pd.DataFrame(
        {
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
    key_dupes = df.duplicated(subset=['Season', 'Team1ID', 'Team2ID']).sum()
    if key_dupes:
        raise ValueError(f'{table_name}: duplicate matchup keys found: {key_dupes}')

    if with_target and not set(df['target'].unique()).issubset({0, 1}):
        raise ValueError(f'{table_name}: target contains values outside 0/1')


def build_gender_pipeline(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = pd.read_csv(RAW_DIR / f'{prefix}RegularSeasonCompactResults.csv')
    tour = pd.read_csv(RAW_DIR / f'{prefix}NCAATourneyCompactResults.csv')
    seeds = pd.read_csv(RAW_DIR / f'{prefix}NCAATourneySeeds.csv')

    team_feats = build_team_season_features(reg)
    seed_table = build_seed_table(seeds)

    train_pairs = make_canonical_tourney_matchups(tour)
    train_feat = attach_pair_features(train_pairs, team_feats, seed_table, include_target=True)
    validate_pair_table(train_feat, with_target=True, table_name=f'{prefix} train')

    return team_feats, train_feat


def build_inference_features(
    sample_sub: pd.DataFrame,
    men_team_feats: pd.DataFrame,
    men_seed_table: pd.DataFrame,
    women_team_feats: pd.DataFrame,
    women_seed_table: pd.DataFrame,
) -> pd.DataFrame:
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

    men_reg = pd.read_csv(RAW_DIR / 'MRegularSeasonCompactResults.csv')
    women_reg = pd.read_csv(RAW_DIR / 'WRegularSeasonCompactResults.csv')
    men_seeds_raw = pd.read_csv(RAW_DIR / 'MNCAATourneySeeds.csv')
    women_seeds_raw = pd.read_csv(RAW_DIR / 'WNCAATourneySeeds.csv')
    men_tour = pd.read_csv(RAW_DIR / 'MNCAATourneyCompactResults.csv')
    women_tour = pd.read_csv(RAW_DIR / 'WNCAATourneyCompactResults.csv')

    men_team_feats = build_team_season_features(men_reg)
    women_team_feats = build_team_season_features(women_reg)
    men_seed_table = build_seed_table(men_seeds_raw)
    women_seed_table = build_seed_table(women_seeds_raw)

    men_train = attach_pair_features(
        make_canonical_tourney_matchups(men_tour),
        men_team_feats,
        men_seed_table,
        include_target=True,
    )
    women_train = attach_pair_features(
        make_canonical_tourney_matchups(women_tour),
        women_team_feats,
        women_seed_table,
        include_target=True,
    )

    validate_pair_table(men_train, with_target=True, table_name='men train')
    validate_pair_table(women_train, with_target=True, table_name='women train')

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
