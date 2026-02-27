# AGENTS.md — Kaggle NCAA March Madness (Men + Women)

## Goal
Build a reproducible pipeline to generate features from historical NCAA results (men + women) and train models to predict March Madness game outcomes for Kaggle submissions.

Primary tasks:
- Load and validate CSVs from `data/raw/`

## Project Layout
- `data/raw/` — Kaggle-provided CSVs (read-only; never overwrite)
- `data/cleaned/` — cleaned/merged tables (generated) or model-ready feature tables (generated)
- `data/submissions/` — `SampleSubmissionStage1.csv` copies + generated submissions
- `notebooks/` — work, where current notebooks life


## Data Files (Kaggle CSVs)
Naming convention:
- `M*` = Men’s-only data
- `W*` = Women’s-only data
- no prefix = shared across both

### Section 1 — Basics
**Teams**
- `MTeams.csv`
  - `TeamID` (1000–1999), `TeamName`, `FirstD1Season`, `LastD1Season`
- `WTeams.csv`
  - `TeamID` (3000–3999), `TeamName`

**Seasons**
- `MSeasons.csv`, `WSeasons.csv`
  - `Season`, `DayZero`, `RegionW/X/Y/Z`
  - Use `DayZero + DayNum` to reconstruct calendar dates if needed.

**Tournament Seeds**
- `MNCAATourneySeeds.csv`, `WNCAATourneySeeds.csv`
  - `Season`, `Seed` (e.g. `W01`, `X16a`), `TeamID`

**Regular Season Results (Compact)**
- `MRegularSeasonCompactResults.csv`, `WRegularSeasonCompactResults.csv`
  - `Season`, `DayNum`, `WTeamID`, `WScore`, `LTeamID`, `LScore`, `WLoc`, `NumOT`
  - Note: `W*` here means “winning”, not “women’s”.

**NCAA Tournament Results (Compact)**
- `MNCAATourneyCompactResults.csv`, `WNCAATourneyCompactResults.csv`
  - Same schema as compact regular season results.

**Submission Template**
- `SampleSubmissionStage1.csv`
  - `ID` = `SSSS_XXXX_YYYY` (Season + low TeamID + high TeamID)
  - `Pred` = probability that `XXXX` beats `YYYY`

### Section 2 — Team Box Scores (Detailed Results)
**Regular Season (Detailed)**
- `MRegularSeasonDetailedResults.csv`, `WRegularSeasonDetailedResults.csv`
  - Same first 8 columns as compact, plus team box score stats.
  - Use for richer features (shooting, rebounding, turnovers, etc.).

**NCAA Tournament (Detailed)**
- `MNCAATourneyDetailedResults.csv`, `WNCAATourneyDetailedResults.csv`
  - Same as detailed regular season, but for tourney games.

### Section 3 — Geography
**Cities Master**
- `Cities.csv`
  - `CityID`, `City`, `State`

**Game Cities**
- `MGameCities.csv`, `WGameCities.csv`
  - `Season`, `DayNum`, `WTeamID`, `LTeamID`, `CRType`, `CityID`
  - `CRType` ∈ {`Regular`, `NCAA`, `Secondary`}

### Section 4 — Public Rankings (Men’s only)
- `MMasseyOrdinals.csv`
  - `Season`, `RankingDayNum`, `SystemName`, `TeamID`, `OrdinalRank`
  - Use to engineer ranking-based features; note system availability varies by year.

### Section 5 — Supplements
**Coaches (Men’s only)**
- `MTeamCoaches.csv`
  - `Season`, `TeamID`, `FirstDayNum`, `LastDayNum`, `CoachName`

**Conferences (Shared)**
- `Conferences.csv`
  - `ConfAbbrev`, `Description`

**Team Conferences**
- `MTeamConferences.csv`, `WTeamConferences.csv`
  - `Season`, `TeamID`, `ConfAbbrev`

**Conference Tourney Games**
- `MConferenceTourneyGames.csv`, `WConferenceTourneyGames.csv`
  - `ConfAbbrev`, `Season`, `DayNum`, `WTeamID`, `LTeamID`

**Secondary Tournament Participation**
- `MSecondaryTourneyTeams.csv`, `WSecondaryTourneyTeams.csv`
  - `Season`, `SecondaryTourney`, `TeamID`

**Secondary Tournament Results (Compact)**
- `MSecondaryTourneyCompactResults.csv`, `WSecondaryTourneyCompactResults.csv`
  - Compact results schema + `SecondaryTourney`

**Team Spellings**
- `MTeamSpellings.csv`, `WTeamSpellings.csv`
  - `TeamNameSpelling`, `TeamID`
  - Useful for mapping external data sources into Kaggle TeamIDs.

**Tournament Bracket Structure**
- `MNCAATourneySlots.csv`, `WNCAATourneySlots.csv`
  - `Season`, `Slot`, `StrongSeed`, `WeakSeed`
- `MNCAATourneySeedRoundSlots.csv` (Men’s only)
  - `Seed`, `GameRound`, `GameSlot`, `EarlyDayNum`, `LateDayNum`

## Reproducibility Rules
- Do not modify files in `data/raw/`.