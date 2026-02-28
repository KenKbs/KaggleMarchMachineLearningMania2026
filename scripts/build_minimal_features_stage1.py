import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

RAW_DIR = Path('data/raw')
CLEAN_DIR = Path('data/cleaned')
ELO_BASE = 1500.0
ELO_K = 20.0
ELO_CARRYOVER = 0.75
EXPORTED_ELO_DROP_COLS = ['EloDiff', 'EloWinProbTeam1']
H2H_WINDOW_SEASONS = 3
H2H_INCLUDE_CURRENT_SEASON_REGULAR = True
H2H_INCLUDE_PRIOR_NCAA = True
H2H_ZERO_MARGIN_IF_NO_WINS = 0.0
H2H_FEATURE_COLS = [
    'H2H_Team1_Wins_3y',
    'H2H_Team2_Wins_3y',
    'H2H_Team1_AvgWinMargin_3y',
    'H2H_Team2_AvgWinMargin_3y',
]
ADVANCED_TEAM_FEATURE_COLS = [
    'Advanced_OffRtg',
    'Advanced_DefRtg',
    'Advanced_eFG',
    'Advanced_TOVPct',
    'Advanced_ORR',
    'Advanced_FTr',
    'Advanced_Pace',
]
LAST10_ADVANCED_TEAM_FEATURE_COLS = [
    'Last10_Advanced_OffRtg',
    'Last10_Advanced_DefRtg',
    'Last10_Advanced_eFG',
    'Last10_Advanced_TOVPct',
    'Last10_Advanced_ORR',
    'Last10_Advanced_FTr',
    'Last10_Advanced_Pace',
]
ADVANCED_DIFF_FEATURE_COLS = [
    'OffRtgDiff',
    'DefRtgDiff',
    'eFGDiff',
    'TOVPctDiff',
    'ORRDiff',
    'FTrDiff',
    'PaceDiff',
    'Last10_OffRtgDiff',
    'Last10_DefRtgDiff',
    'Last10_eFGDiff',
    'Last10_TOVPctDiff',
    'Last10_ORRDiff',
    'Last10_FTrDiff',
    'Last10_PaceDiff',
]


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


def elo_expected_score(rating_a: float, rating_b: float) -> float:
    """Return Elo expected score for team A against team B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def assert_elo_examples() -> None:
    """Sanity checks for Elo behavior."""
    p_equal = elo_expected_score(1500.0, 1500.0)
    assert abs(p_equal - 0.5) < 1e-12

    p_favorite = elo_expected_score(1700.0, 1300.0)
    p_underdog = elo_expected_score(1300.0, 1700.0)
    assert p_favorite > 0.5
    assert p_underdog < 0.5
    assert abs((p_favorite + p_underdog) - 1.0) < 1e-12


def _season_start_rating(prev_end: float | None) -> float:
    """Compute season start Elo with soft carryover."""
    if prev_end is None:
        return ELO_BASE
    return ELO_CARRYOVER * prev_end + (1.0 - ELO_CARRYOVER) * ELO_BASE


def build_elo_tables(
    reg_compact: pd.DataFrame,
    tour_compact: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build Elo game timeline plus regular-season and season-end snapshots."""
    key_cols = ['Season', 'DayNum', 'WTeamID', 'LTeamID']
    reg_games = reg_compact[key_cols].copy()
    reg_games['GameType'] = 'Regular'
    tour_games = tour_compact[key_cols].copy()
    tour_games['GameType'] = 'NCAA'

    all_games = pd.concat([reg_games, tour_games], ignore_index=True)
    all_games = all_games.sort_values(
        ['Season', 'DayNum', 'WTeamID', 'LTeamID', 'GameType']
    ).reset_index(drop=True)

    key_dupes = all_games.duplicated(subset=key_cols).sum()
    if key_dupes:
        raise ValueError(f'elo games: duplicate game keys found: {key_dupes}')

    elo_records: list[dict[str, float | int | str]] = []
    regular_end_records: list[dict[str, float | int]] = []
    season_end_records: list[dict[str, float | int]] = []

    prev_end_ratings: dict[int, float] = {}
    seasons = sorted(all_games['Season'].unique())

    for season in seasons:
        season_games = all_games[all_games['Season'] == season].copy()
        season_games = season_games.sort_values(
            ['DayNum', 'WTeamID', 'LTeamID', 'GameType']
        )

        teams = sorted(set(season_games['WTeamID']) | set(season_games['LTeamID']))
        ratings: dict[int, float] = {
            int(team_id): _season_start_rating(prev_end_ratings.get(int(team_id)))
            for team_id in teams
        }

        regular_snapshot_captured = False
        regular_end_ratings: dict[int, float] | None = None

        for row in season_games.itertuples(index=False):
            if row.GameType != 'Regular' and not regular_snapshot_captured:
                regular_end_ratings = ratings.copy()
                regular_snapshot_captured = True

            winner_id = int(row.WTeamID)
            loser_id = int(row.LTeamID)
            winner_rating = ratings[winner_id]
            loser_rating = ratings[loser_id]

            expected_winner = elo_expected_score(winner_rating, loser_rating)
            expected_loser = 1.0 - expected_winner
            elo_records.append(
                {
                    'Season': int(row.Season),
                    'DayNum': int(row.DayNum),
                    'WTeamID': winner_id,
                    'LTeamID': loser_id,
                    'GameType': str(row.GameType),
                    'W_EloPregame': winner_rating,
                    'L_EloPregame': loser_rating,
                    'W_Expected': expected_winner,
                    'L_Expected': expected_loser,
                }
            )

            ratings[winner_id] = winner_rating + ELO_K * (1.0 - expected_winner)
            ratings[loser_id] = loser_rating + ELO_K * (0.0 - expected_loser)

        if not regular_snapshot_captured:
            regular_end_ratings = ratings.copy()

        assert regular_end_ratings is not None
        for team_id, rating in regular_end_ratings.items():
            regular_end_records.append(
                {'Season': int(season), 'TeamID': int(team_id), 'EloPregame': float(rating)}
            )
        for team_id, rating in ratings.items():
            season_end_records.append(
                {'Season': int(season), 'TeamID': int(team_id), 'EloSeasonEnd': float(rating)}
            )

        prev_end_ratings = ratings.copy()

    elo_games = pd.DataFrame(elo_records)
    regular_end_snapshot = pd.DataFrame(regular_end_records)
    season_end_snapshot = pd.DataFrame(season_end_records)

    reg_dupes = regular_end_snapshot.duplicated(subset=['Season', 'TeamID']).sum()
    if reg_dupes:
        raise ValueError(f'regular end elo snapshot: duplicate (Season, TeamID) rows found: {reg_dupes}')
    end_dupes = season_end_snapshot.duplicated(subset=['Season', 'TeamID']).sum()
    if end_dupes:
        raise ValueError(f'season end elo snapshot: duplicate (Season, TeamID) rows found: {end_dupes}')

    return elo_games, regular_end_snapshot, season_end_snapshot


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


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise safe divide with 0.0 fallback when denominator is <= 0."""
    den = den.astype(float)
    num = num.astype(float)
    out = pd.Series(np.zeros(len(num), dtype=float), index=num.index)
    mask = den > 0
    out.loc[mask] = num.loc[mask] / den.loc[mask]
    return out


def build_team_game_rows_detailed(reg_detailed: pd.DataFrame) -> pd.DataFrame:
    """Turn detailed regular-season games into one row per team-game with team/opponent stats."""
    w_rows = reg_detailed[
        [
            'Season', 'DayNum', 'WTeamID', 'LTeamID', 'WScore', 'LScore',
            'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO',
            'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR',
            'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
        ]
    ].copy()
    w_rows = w_rows.rename(
        columns={
            'WTeamID': 'TeamID',
            'LTeamID': 'OppTeamID',
            'WScore': 'TeamScore',
            'LScore': 'OppScore',
            'WFGM': 'TeamFGM',
            'WFGA': 'TeamFGA',
            'WFGM3': 'TeamFGM3',
            'WFGA3': 'TeamFGA3',
            'WFTM': 'TeamFTM',
            'WFTA': 'TeamFTA',
            'WOR': 'TeamOR',
            'WDR': 'TeamDR',
            'WAst': 'TeamAst',
            'WTO': 'TeamTO',
            'WStl': 'TeamStl',
            'WBlk': 'TeamBlk',
            'WPF': 'TeamPF',
            'LFGM': 'OppFGM',
            'LFGA': 'OppFGA',
            'LFGM3': 'OppFGM3',
            'LFGA3': 'OppFGA3',
            'LFTM': 'OppFTM',
            'LFTA': 'OppFTA',
            'LOR': 'OppOR',
            'LDR': 'OppDR',
            'LAst': 'OppAst',
            'LTO': 'OppTO',
            'LStl': 'OppStl',
            'LBlk': 'OppBlk',
            'LPF': 'OppPF',
        }
    )

    l_rows = reg_detailed[
        [
            'Season', 'DayNum', 'LTeamID', 'WTeamID', 'LScore', 'WScore',
            'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
            'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR',
            'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
        ]
    ].copy()
    l_rows = l_rows.rename(
        columns={
            'LTeamID': 'TeamID',
            'WTeamID': 'OppTeamID',
            'LScore': 'TeamScore',
            'WScore': 'OppScore',
            'LFGM': 'TeamFGM',
            'LFGA': 'TeamFGA',
            'LFGM3': 'TeamFGM3',
            'LFGA3': 'TeamFGA3',
            'LFTM': 'TeamFTM',
            'LFTA': 'TeamFTA',
            'LOR': 'TeamOR',
            'LDR': 'TeamDR',
            'LAst': 'TeamAst',
            'LTO': 'TeamTO',
            'LStl': 'TeamStl',
            'LBlk': 'TeamBlk',
            'LPF': 'TeamPF',
            'WFGM': 'OppFGM',
            'WFGA': 'OppFGA',
            'WFGM3': 'OppFGM3',
            'WFGA3': 'OppFGA3',
            'WFTM': 'OppFTM',
            'WFTA': 'OppFTA',
            'WOR': 'OppOR',
            'WDR': 'OppDR',
            'WAst': 'OppAst',
            'WTO': 'OppTO',
            'WStl': 'OppStl',
            'WBlk': 'OppBlk',
            'WPF': 'OppPF',
        }
    )

    rows = pd.concat([w_rows, l_rows], ignore_index=True)
    rows['TeamPoss'] = rows['TeamFGA'] - rows['TeamOR'] + rows['TeamTO'] + 0.475 * rows['TeamFTA']
    rows['OppPoss'] = rows['OppFGA'] - rows['OppOR'] + rows['OppTO'] + 0.475 * rows['OppFTA']
    return rows


def compute_advanced_metrics_from_agg(
    agg: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Compute advanced box-score metrics from aggregated numerator/denominator sums."""
    out = agg[['Season', 'TeamID']].copy()
    out[f'{prefix}OffRtg'] = 100.0 * safe_divide(agg['TeamScore'], agg['TeamPoss'])
    out[f'{prefix}DefRtg'] = 100.0 * safe_divide(agg['OppScore'], agg['OppPoss'])
    out[f'{prefix}eFG'] = safe_divide(agg['TeamFGM'] + 0.5 * agg['TeamFGM3'], agg['TeamFGA'])
    out[f'{prefix}TOVPct'] = safe_divide(
        agg['TeamTO'],
        agg['TeamFGA'] + 0.475 * agg['TeamFTA'] + agg['TeamTO'],
    )
    out[f'{prefix}ORR'] = safe_divide(agg['TeamOR'], agg['TeamOR'] + agg['OppDR'])
    out[f'{prefix}FTr'] = safe_divide(agg['TeamFTA'], agg['TeamFGA'])
    out[f'{prefix}Pace'] = safe_divide(agg['TeamPoss'], agg['Games'])
    return out


def build_team_season_features(reg_compact: pd.DataFrame, reg_detailed: pd.DataFrame) -> pd.DataFrame:
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

    detailed_rows = build_team_game_rows_detailed(reg_detailed)
    detailed_agg = (
        detailed_rows.groupby(['Season', 'TeamID'], as_index=False)
        .agg(
            Games=('TeamScore', 'size'),
            TeamScore=('TeamScore', 'sum'),
            OppScore=('OppScore', 'sum'),
            TeamFGM=('TeamFGM', 'sum'),
            TeamFGA=('TeamFGA', 'sum'),
            TeamFGM3=('TeamFGM3', 'sum'),
            TeamFTA=('TeamFTA', 'sum'),
            TeamTO=('TeamTO', 'sum'),
            TeamOR=('TeamOR', 'sum'),
            OppDR=('OppDR', 'sum'),
            TeamPoss=('TeamPoss', 'sum'),
            OppPoss=('OppPoss', 'sum'),
        )
    )
    advanced = compute_advanced_metrics_from_agg(detailed_agg, prefix='Advanced_')

    detailed_last10 = (
        detailed_rows.sort_values(['Season', 'TeamID', 'DayNum'])
        .groupby(['Season', 'TeamID'], as_index=False)
        .tail(10)
    )
    detailed_last10_agg = (
        detailed_last10.groupby(['Season', 'TeamID'], as_index=False)
        .agg(
            Games=('TeamScore', 'size'),
            TeamScore=('TeamScore', 'sum'),
            OppScore=('OppScore', 'sum'),
            TeamFGM=('TeamFGM', 'sum'),
            TeamFGA=('TeamFGA', 'sum'),
            TeamFGM3=('TeamFGM3', 'sum'),
            TeamFTA=('TeamFTA', 'sum'),
            TeamTO=('TeamTO', 'sum'),
            TeamOR=('TeamOR', 'sum'),
            OppDR=('OppDR', 'sum'),
            TeamPoss=('TeamPoss', 'sum'),
            OppPoss=('OppPoss', 'sum'),
        )
    )
    advanced_last10 = compute_advanced_metrics_from_agg(
        detailed_last10_agg,
        prefix='Last10_Advanced_',
    )

    feats = feats.merge(advanced, on=['Season', 'TeamID'], how='left')
    feats = feats.merge(advanced_last10, on=['Season', 'TeamID'], how='left')
    feats = feats[[
        'Season',
        'TeamID',
        'RegularSeason_WinPct',
        'RegularSeason_PointMarginPerGame',
        'Last10_WinPct',
        'Last10PointMargin',
        *ADVANCED_TEAM_FEATURE_COLS,
        *LAST10_ADVANCED_TEAM_FEATURE_COLS,
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


def make_canonical_tourney_matchups(
    tourney_compact: pd.DataFrame,
    tourney_elo: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert hiistorical tournament results into a consistent training dataset format

    Args:
        tourney_compact (pd.DataFrame): usualyl dataframe generated from MNCAATourneyCompactResults.csv

    Returns:
        pd.DataFrame: Team1ID, Team2ID, target (= 1 if Team1 won, 0 if Team 2 won)
    """
    key_cols = ['Season', 'DayNum', 'WTeamID', 'LTeamID']
    out = tourney_compact[key_cols].copy()

    if tourney_elo is not None:
        required = [*key_cols, 'W_EloPregame', 'L_EloPregame']
        missing = [c for c in required if c not in tourney_elo.columns]
        if missing:
            raise ValueError(f'tourney elo: missing required columns: {missing}')

        elo_dupes = tourney_elo.duplicated(subset=key_cols).sum()
        if elo_dupes:
            raise ValueError(f'tourney elo: duplicate game keys found: {elo_dupes}')

        out = out.merge(tourney_elo[required], on=key_cols, how='left')
        missing_elo = out['W_EloPregame'].isna() | out['L_EloPregame'].isna()
        if missing_elo.any():
            raise ValueError(
                f'tourney elo: missing Elo matches for {int(missing_elo.sum())} tournament rows'
            )

    out['Team1ID'] = out[['WTeamID', 'LTeamID']].min(axis=1)
    out['Team2ID'] = out[['WTeamID', 'LTeamID']].max(axis=1)
    out['target'] = (out['WTeamID'] == out['Team1ID']).astype(int)

    if tourney_elo is not None:
        out['Team1_EloPregame'] = np.where(
            out['WTeamID'] == out['Team1ID'],
            out['W_EloPregame'],
            out['L_EloPregame'],
        )
        out['Team2_EloPregame'] = np.where(
            out['WTeamID'] == out['Team2ID'],
            out['W_EloPregame'],
            out['L_EloPregame'],
        )
        return out[
            [
                'Season',
                'Team1ID',
                'Team2ID',
                'target',
                'Team1_EloPregame',
                'Team2_EloPregame',
            ]
        ].copy()

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


def build_canonical_h2h_history(
    reg_compact: pd.DataFrame,
    tour_compact: pd.DataFrame,
    include_prior_ncaa: bool,
) -> pd.DataFrame:
    """Create canonical head-to-head game history rows across regular and optional NCAA games."""
    reg = reg_compact[
        ['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore']
    ].copy()
    reg['Source'] = 'Regular'

    frames = [reg]
    if include_prior_ncaa:
        tour = tour_compact[
            ['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore']
        ].copy()
        tour['Source'] = 'NCAA'
        frames.append(tour)

    games = pd.concat(frames, ignore_index=True)
    games['Team1ID'] = games[['WTeamID', 'LTeamID']].min(axis=1).astype(int)
    games['Team2ID'] = games[['WTeamID', 'LTeamID']].max(axis=1).astype(int)
    games['Team1Won'] = (games['WTeamID'] == games['Team1ID']).astype(int)
    games['Team1Margin'] = np.where(
        games['Team1Won'] == 1,
        games['WScore'] - games['LScore'],
        games['LScore'] - games['WScore'],
    ).astype(float)
    return games[
        ['Season', 'Team1ID', 'Team2ID', 'Source', 'Team1Won', 'Team1Margin']
    ].copy()


def build_h2h_features(
    pair_df: pd.DataFrame,
    reg_compact: pd.DataFrame,
    tour_compact: pd.DataFrame,
    window_seasons: int = H2H_WINDOW_SEASONS,
) -> pd.DataFrame:
    """Build rolling head-to-head features for each matchup row."""
    required = ['Season', 'Team1ID', 'Team2ID']
    missing = [c for c in required if c not in pair_df.columns]
    if missing:
        raise ValueError(f'build_h2h_features: pair_df missing required columns: {missing}')

    history = build_canonical_h2h_history(
        reg_compact=reg_compact,
        tour_compact=tour_compact,
        include_prior_ncaa=H2H_INCLUDE_PRIOR_NCAA,
    )
    history['Team2Won'] = 1 - history['Team1Won']
    history['Team1WinMargin'] = np.where(history['Team1Won'] == 1, history['Team1Margin'], 0.0)
    history['Team2WinMargin'] = np.where(history['Team2Won'] == 1, -history['Team1Margin'], 0.0)

    regular_year = (
        history[history['Source'] == 'Regular']
        .groupby(['Season', 'Team1ID', 'Team2ID'], as_index=False)
        .agg(
            Team1Wins=('Team1Won', 'sum'),
            Team2Wins=('Team2Won', 'sum'),
            Team1WinMarginSum=('Team1WinMargin', 'sum'),
            Team2WinMarginSum=('Team2WinMargin', 'sum'),
        )
    )
    ncaa_year = (
        history[history['Source'] == 'NCAA']
        .groupby(['Season', 'Team1ID', 'Team2ID'], as_index=False)
        .agg(
            Team1Wins=('Team1Won', 'sum'),
            Team2Wins=('Team2Won', 'sum'),
            Team1WinMarginSum=('Team1WinMargin', 'sum'),
            Team2WinMarginSum=('Team2WinMargin', 'sum'),
        )
    )

    rows: list[pd.DataFrame] = []
    for season in sorted(pair_df['Season'].unique()):
        season_pairs = pair_df[pair_df['Season'] == season][['Season', 'Team1ID', 'Team2ID']].copy()
        hist_start = int(season) - window_seasons + 1

        reg_hist = regular_year[
            (regular_year['Season'] >= hist_start)
            & (regular_year['Season'] <= int(season))
        ][['Team1ID', 'Team2ID', 'Team1Wins', 'Team2Wins', 'Team1WinMarginSum', 'Team2WinMarginSum']]

        hist_frames = [reg_hist]
        if H2H_INCLUDE_PRIOR_NCAA and not ncaa_year.empty:
            ncaa_hist = ncaa_year[
                (ncaa_year['Season'] >= hist_start)
                & (ncaa_year['Season'] <= int(season) - 1)
            ][['Team1ID', 'Team2ID', 'Team1Wins', 'Team2Wins', 'Team1WinMarginSum', 'Team2WinMarginSum']]
            hist_frames.append(ncaa_hist)

        hist = pd.concat(hist_frames, ignore_index=True)
        if hist.empty:
            season_pairs['H2H_Team1_Wins_3y'] = 0
            season_pairs['H2H_Team2_Wins_3y'] = 0
            season_pairs['H2H_Team1_AvgWinMargin_3y'] = H2H_ZERO_MARGIN_IF_NO_WINS
            season_pairs['H2H_Team2_AvgWinMargin_3y'] = H2H_ZERO_MARGIN_IF_NO_WINS
            rows.append(season_pairs)
            continue

        hist = (
            hist.groupby(['Team1ID', 'Team2ID'], as_index=False)
            .agg(
                Team1Wins=('Team1Wins', 'sum'),
                Team2Wins=('Team2Wins', 'sum'),
                Team1WinMarginSum=('Team1WinMarginSum', 'sum'),
                Team2WinMarginSum=('Team2WinMarginSum', 'sum'),
            )
        )
        merged = season_pairs.merge(hist, on=['Team1ID', 'Team2ID'], how='left')
        for col in ['Team1Wins', 'Team2Wins', 'Team1WinMarginSum', 'Team2WinMarginSum']:
            merged[col] = merged[col].fillna(0.0)

        merged['H2H_Team1_Wins_3y'] = merged['Team1Wins'].astype(int)
        merged['H2H_Team2_Wins_3y'] = merged['Team2Wins'].astype(int)
        merged['H2H_Team1_AvgWinMargin_3y'] = np.where(
            merged['Team1Wins'] > 0,
            merged['Team1WinMarginSum'] / merged['Team1Wins'],
            H2H_ZERO_MARGIN_IF_NO_WINS,
        )
        merged['H2H_Team2_AvgWinMargin_3y'] = np.where(
            merged['Team2Wins'] > 0,
            merged['Team2WinMarginSum'] / merged['Team2Wins'],
            H2H_ZERO_MARGIN_IF_NO_WINS,
        )
        rows.append(merged[['Season', 'Team1ID', 'Team2ID', *H2H_FEATURE_COLS]])

    out = pd.concat(rows, ignore_index=True)
    return out[['Season', 'Team1ID', 'Team2ID', *H2H_FEATURE_COLS]].copy()


def attach_pair_features(
    pair_df: pd.DataFrame,
    team_feats: pd.DataFrame,
    seeds: pd.DataFrame,
    include_target: bool,
    elo_snapshot: pd.DataFrame | None = None,
    h2h_features: pd.DataFrame | None = None,
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
            'Advanced_OffRtg': 'Team1_Advanced_OffRtg',
            'Advanced_DefRtg': 'Team1_Advanced_DefRtg',
            'Advanced_eFG': 'Team1_Advanced_eFG',
            'Advanced_TOVPct': 'Team1_Advanced_TOVPct',
            'Advanced_ORR': 'Team1_Advanced_ORR',
            'Advanced_FTr': 'Team1_Advanced_FTr',
            'Advanced_Pace': 'Team1_Advanced_Pace',
            'Last10_Advanced_OffRtg': 'Team1_Last10_Advanced_OffRtg',
            'Last10_Advanced_DefRtg': 'Team1_Last10_Advanced_DefRtg',
            'Last10_Advanced_eFG': 'Team1_Last10_Advanced_eFG',
            'Last10_Advanced_TOVPct': 'Team1_Last10_Advanced_TOVPct',
            'Last10_Advanced_ORR': 'Team1_Last10_Advanced_ORR',
            'Last10_Advanced_FTr': 'Team1_Last10_Advanced_FTr',
            'Last10_Advanced_Pace': 'Team1_Last10_Advanced_Pace',
        }
    )
    t2 = team_feats.rename(
        columns={
            'TeamID': 'Team2ID',
            'RegularSeason_WinPct': 'Team2_WinPct',
            'RegularSeason_PointMarginPerGame': 'Team2_PointMarginPerGame',
            'Last10_WinPct': 'Team2_Last10_WinPct',
            'Last10PointMargin': 'Team2_Last10_PointMargin',
            'Advanced_OffRtg': 'Team2_Advanced_OffRtg',
            'Advanced_DefRtg': 'Team2_Advanced_DefRtg',
            'Advanced_eFG': 'Team2_Advanced_eFG',
            'Advanced_TOVPct': 'Team2_Advanced_TOVPct',
            'Advanced_ORR': 'Team2_Advanced_ORR',
            'Advanced_FTr': 'Team2_Advanced_FTr',
            'Advanced_Pace': 'Team2_Advanced_Pace',
            'Last10_Advanced_OffRtg': 'Team2_Last10_Advanced_OffRtg',
            'Last10_Advanced_DefRtg': 'Team2_Last10_Advanced_DefRtg',
            'Last10_Advanced_eFG': 'Team2_Last10_Advanced_eFG',
            'Last10_Advanced_TOVPct': 'Team2_Last10_Advanced_TOVPct',
            'Last10_Advanced_ORR': 'Team2_Last10_Advanced_ORR',
            'Last10_Advanced_FTr': 'Team2_Last10_Advanced_FTr',
            'Last10_Advanced_Pace': 'Team2_Last10_Advanced_Pace',
        }
    )

    s1 = seeds.rename(columns={'TeamID': 'Team1ID', 'SeedNum': 'Team1_SeedNum'})
    s2 = seeds.rename(columns={'TeamID': 'Team2ID', 'SeedNum': 'Team2_SeedNum'})

    merged = pair_df.merge(t1, on=['Season', 'Team1ID'], how='left')
    merged = merged.merge(t2, on=['Season', 'Team2ID'], how='left')
    merged = merged.merge(s1, on=['Season', 'Team1ID'], how='left')
    merged = merged.merge(s2, on=['Season', 'Team2ID'], how='left')

    if elo_snapshot is not None:
        e1 = elo_snapshot.rename(columns={'TeamID': 'Team1ID', 'EloPregame': 'Team1_EloPregame'})
        e2 = elo_snapshot.rename(columns={'TeamID': 'Team2ID', 'EloPregame': 'Team2_EloPregame'})
        merged = merged.merge(e1, on=['Season', 'Team1ID'], how='left')
        merged = merged.merge(e2, on=['Season', 'Team2ID'], how='left')
    if h2h_features is not None:
        merged = merged.merge(
            h2h_features[['Season', 'Team1ID', 'Team2ID', *H2H_FEATURE_COLS]],
            on=['Season', 'Team1ID', 'Team2ID'],
            how='left',
        )
    else:
        for col in H2H_FEATURE_COLS:
            merged[col] = 0.0

    required_elo_cols = ['Team1_EloPregame', 'Team2_EloPregame']
    missing_elo_cols = [c for c in required_elo_cols if c not in merged.columns]
    if missing_elo_cols:
        raise ValueError(
            'attach_pair_features: missing Elo columns. '
            'Provide `elo_snapshot` or pass pair_df with Team1_EloPregame/Team2_EloPregame.'
        )

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

    advanced_raw_cols = [
        'Team1_Advanced_OffRtg', 'Team1_Advanced_DefRtg', 'Team1_Advanced_eFG',
        'Team1_Advanced_TOVPct', 'Team1_Advanced_ORR', 'Team1_Advanced_FTr', 'Team1_Advanced_Pace',
        'Team2_Advanced_OffRtg', 'Team2_Advanced_DefRtg', 'Team2_Advanced_eFG',
        'Team2_Advanced_TOVPct', 'Team2_Advanced_ORR', 'Team2_Advanced_FTr', 'Team2_Advanced_Pace',
        'Team1_Last10_Advanced_OffRtg', 'Team1_Last10_Advanced_DefRtg', 'Team1_Last10_Advanced_eFG',
        'Team1_Last10_Advanced_TOVPct', 'Team1_Last10_Advanced_ORR', 'Team1_Last10_Advanced_FTr',
        'Team1_Last10_Advanced_Pace',
        'Team2_Last10_Advanced_OffRtg', 'Team2_Last10_Advanced_DefRtg', 'Team2_Last10_Advanced_eFG',
        'Team2_Last10_Advanced_TOVPct', 'Team2_Last10_Advanced_ORR', 'Team2_Last10_Advanced_FTr',
        'Team2_Last10_Advanced_Pace',
    ]
    for col in advanced_raw_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    merged['OffRtgDiff'] = merged['Team1_Advanced_OffRtg'] - merged['Team2_Advanced_OffRtg']
    merged['DefRtgDiff'] = merged['Team1_Advanced_DefRtg'] - merged['Team2_Advanced_DefRtg']
    merged['eFGDiff'] = merged['Team1_Advanced_eFG'] - merged['Team2_Advanced_eFG']
    merged['TOVPctDiff'] = merged['Team1_Advanced_TOVPct'] - merged['Team2_Advanced_TOVPct']
    merged['ORRDiff'] = merged['Team1_Advanced_ORR'] - merged['Team2_Advanced_ORR']
    merged['FTrDiff'] = merged['Team1_Advanced_FTr'] - merged['Team2_Advanced_FTr']
    merged['PaceDiff'] = merged['Team1_Advanced_Pace'] - merged['Team2_Advanced_Pace']
    merged['Last10_OffRtgDiff'] = (
        merged['Team1_Last10_Advanced_OffRtg'] - merged['Team2_Last10_Advanced_OffRtg']
    )
    merged['Last10_DefRtgDiff'] = (
        merged['Team1_Last10_Advanced_DefRtg'] - merged['Team2_Last10_Advanced_DefRtg']
    )
    merged['Last10_eFGDiff'] = (
        merged['Team1_Last10_Advanced_eFG'] - merged['Team2_Last10_Advanced_eFG']
    )
    merged['Last10_TOVPctDiff'] = (
        merged['Team1_Last10_Advanced_TOVPct'] - merged['Team2_Last10_Advanced_TOVPct']
    )
    merged['Last10_ORRDiff'] = (
        merged['Team1_Last10_Advanced_ORR'] - merged['Team2_Last10_Advanced_ORR']
    )
    merged['Last10_FTrDiff'] = (
        merged['Team1_Last10_Advanced_FTr'] - merged['Team2_Last10_Advanced_FTr']
    )
    merged['Last10_PaceDiff'] = (
        merged['Team1_Last10_Advanced_Pace'] - merged['Team2_Last10_Advanced_Pace']
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
        'Team1_EloPregame',
        'Team2_EloPregame',
        *ADVANCED_DIFF_FEATURE_COLS,
        *H2H_FEATURE_COLS,
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

    h2h_present = [c for c in H2H_FEATURE_COLS if c in df.columns]
    if h2h_present:
        missing_h2h = [c for c in H2H_FEATURE_COLS if c not in df.columns]
        if missing_h2h:
            raise ValueError(f'{table_name}: missing H2H columns: {missing_h2h}')
        if df[H2H_FEATURE_COLS].isna().any().any():
            missing_count = int(df[H2H_FEATURE_COLS].isna().sum().sum())
            raise ValueError(f'{table_name}: H2H feature columns contain NaN values: {missing_count}')
        win_cols = ['H2H_Team1_Wins_3y', 'H2H_Team2_Wins_3y']
        if (df[win_cols] < 0).any().any():
            raise ValueError(f'{table_name}: H2H win-count features contain negative values')
        non_int_mask = (df[win_cols] % 1 != 0).any().any()
        if non_int_mask:
            raise ValueError(f'{table_name}: H2H win-count features must be integer-valued')
        margin_cols = ['H2H_Team1_AvgWinMargin_3y', 'H2H_Team2_AvgWinMargin_3y']
        if (df[margin_cols] < 0).any().any():
            raise ValueError(f'{table_name}: H2H average win-margin features contain negative values')

    adv_present = [c for c in ADVANCED_DIFF_FEATURE_COLS if c in df.columns]
    if adv_present:
        missing_adv = [c for c in ADVANCED_DIFF_FEATURE_COLS if c not in df.columns]
        if missing_adv:
            raise ValueError(f'{table_name}: missing advanced diff columns: {missing_adv}')
        if df[ADVANCED_DIFF_FEATURE_COLS].isna().any().any():
            missing_count = int(df[ADVANCED_DIFF_FEATURE_COLS].isna().sum().sum())
            raise ValueError(f'{table_name}: advanced diff columns contain NaN values: {missing_count}')
        if not np.isfinite(df[ADVANCED_DIFF_FEATURE_COLS].to_numpy()).all():
            raise ValueError(f'{table_name}: advanced diff columns contain non-finite values')


def validate_elo_feature_table(df: pd.DataFrame, table_name: str) -> None:
    """Validate Elo feature columns are present and well-formed."""
    required = ['Team1_EloPregame', 'Team2_EloPregame']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f'{table_name}: missing Elo feature columns: {missing_cols}')

    if df[required].isna().any().any():
        missing_count = int(df[required].isna().sum().sum())
        raise ValueError(f'{table_name}: Elo feature columns contain NaN values: {missing_count}')

def fill_missing_elo_from_carryover(
    pair_feats: pd.DataFrame,
    season_end_elo: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing Elo snapshot values using prior-season carryover."""
    out = pair_feats.copy()
    season_end_lookup = (
        season_end_elo.set_index(['Season', 'TeamID'])['EloSeasonEnd'].to_dict()
        if not season_end_elo.empty
        else {}
    )

    def resolve_rating(season: int, team_id: int) -> float:
        prev_end = season_end_lookup.get((int(season) - 1, int(team_id)))
        return _season_start_rating(prev_end)

    for team_col, elo_col in [('Team1ID', 'Team1_EloPregame'), ('Team2ID', 'Team2_EloPregame')]:
        miss_mask = out[elo_col].isna()
        if miss_mask.any():
            replacements = [
                resolve_rating(season, team_id)
                for season, team_id in zip(out.loc[miss_mask, 'Season'], out.loc[miss_mask, team_col])
            ]
            out.loc[miss_mask, elo_col] = replacements

    return out


GenderPrefix = Literal["M", "W"] # Gender prefix for function definition
def build_gender_pipeline(
    prefix: GenderPrefix,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            team features, seed table, regular end Elo table, season end Elo table, and train features
    """
    reg = pd.read_csv(RAW_DIR / f'{prefix}RegularSeasonCompactResults.csv')
    reg_detailed = pd.read_csv(RAW_DIR / f'{prefix}RegularSeasonDetailedResults.csv')
    tour = pd.read_csv(RAW_DIR / f'{prefix}NCAATourneyCompactResults.csv')
    seeds = pd.read_csv(RAW_DIR / f'{prefix}NCAATourneySeeds.csv')

    elo_games, regular_end_elo, season_end_elo = build_elo_tables(reg, tour)
    tourney_elo = elo_games[elo_games['GameType'] == 'NCAA'].copy()

    team_feats = build_team_season_features(reg, reg_detailed)
    seed_table = build_seed_table(seeds)

    train_pairs = make_canonical_tourney_matchups(tour, tourney_elo=tourney_elo)
    train_h2h = build_h2h_features(
        train_pairs,
        reg_compact=reg,
        tour_compact=tour,
        window_seasons=H2H_WINDOW_SEASONS,
    )
    train_feat = attach_pair_features(
        train_pairs,
        team_feats,
        seed_table,
        include_target=True,
        h2h_features=train_h2h,
    )
    train_feat = fill_missing_elo_from_carryover(train_feat, season_end_elo)
    validate_pair_table(train_feat, with_target=True, table_name=f'{prefix} train')
    validate_elo_feature_table(train_feat, table_name=f'{prefix} train')
    train_feat = train_feat.drop(columns=EXPORTED_ELO_DROP_COLS, errors='ignore')

    return team_feats, seed_table, regular_end_elo, season_end_elo, train_feat


def build_inference_features(
    sample_sub: pd.DataFrame,
    men_team_feats: pd.DataFrame,
    men_seed_table: pd.DataFrame,
    men_regular_end_elo: pd.DataFrame,
    men_season_end_elo: pd.DataFrame,
    women_team_feats: pd.DataFrame,
    women_seed_table: pd.DataFrame,
    women_regular_end_elo: pd.DataFrame,
    women_season_end_elo: pd.DataFrame,
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
    men_reg = pd.read_csv(RAW_DIR / 'MRegularSeasonCompactResults.csv')
    men_tour = pd.read_csv(RAW_DIR / 'MNCAATourneyCompactResults.csv')
    women_reg = pd.read_csv(RAW_DIR / 'WRegularSeasonCompactResults.csv')
    women_tour = pd.read_csv(RAW_DIR / 'WNCAATourneyCompactResults.csv')
    men_h2h = build_h2h_features(
        men_pairs,
        reg_compact=men_reg,
        tour_compact=men_tour,
        window_seasons=H2H_WINDOW_SEASONS,
    )
    women_h2h = build_h2h_features(
        women_pairs,
        reg_compact=women_reg,
        tour_compact=women_tour,
        window_seasons=H2H_WINDOW_SEASONS,
    )

    men_feat = attach_pair_features(
        men_pairs,
        men_team_feats,
        men_seed_table,
        include_target=False,
        elo_snapshot=men_regular_end_elo,
        h2h_features=men_h2h,
    )
    women_feat = attach_pair_features(
        women_pairs,
        women_team_feats,
        women_seed_table,
        include_target=False,
        elo_snapshot=women_regular_end_elo,
        h2h_features=women_h2h,
    )
    men_feat = fill_missing_elo_from_carryover(men_feat, men_season_end_elo)
    women_feat = fill_missing_elo_from_carryover(women_feat, women_season_end_elo)

    out = pd.concat([men_feat, women_feat], ignore_index=True)
    out = out.sort_values(['Season', 'Team1ID', 'Team2ID']).reset_index(drop=True)
    validate_pair_table(out, with_target=False, table_name='stage1 inference')
    validate_elo_feature_table(out, table_name='stage1 inference')
    out = out.drop(columns=EXPORTED_ELO_DROP_COLS, errors='ignore')

    if len(out) != len(sample_sub):
        raise ValueError('stage1 inference: row count mismatch against sample submission')

    return out


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    assert_seed_parser_examples()
    assert_elo_examples()

    men_team_feats, men_seed_table, men_regular_end_elo, men_season_end_elo, men_train = build_gender_pipeline('M')
    women_team_feats, women_seed_table, women_regular_end_elo, women_season_end_elo, women_train = build_gender_pipeline('W')

    sample_sub = pd.read_csv(RAW_DIR / 'SampleSubmissionStage1.csv')
    inference = build_inference_features(
        sample_sub,
        men_team_feats,
        men_seed_table,
        men_regular_end_elo,
        men_season_end_elo,
        women_team_feats,
        women_seed_table,
        women_regular_end_elo,
        women_season_end_elo,
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
