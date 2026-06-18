# Spotify Top Songs Analysis — Python

**Author:** Rafael Cruz-Camacho
**Tools:** Python — pandas, matplotlib, seaborn (VS Code)
**Dataset:** Spotify Tracks Dataset — 114,000 tracks across 113 genres

## Objective
Analyze what audio features make songs popular on Spotify —
danceability, energy, tempo, and more — across 113 genres.

## Key Questions
1. Which genres are most popular on average?
2. Which audio features correlate most with popularity?
3. Is there a relationship between energy and danceability?
4. Do explicit tracks perform better than clean ones?
5. What do the top 10 most popular tracks have in common?

## How to Run
```bash
cd spotify-analysis-python
python notebooks/analysis.py
```
Charts saved to `outputs/`. Clean CSV saved to `data/clean/`.

## Project Structure
```
spotify-analysis-python/
├── data/
│   ├── raw/     dataset.csv
│   └── clean/   spotify_clean.csv
├── notebooks/
│   └── analysis.py
├── outputs/
│   ├── 01_top_genres_popularity.png
│   ├── 02_feature_correlations.png
│   ├── 03_energy_vs_danceability.png
│   └── 04_explicit_vs_clean.png
├── reports/
│   └── findings.md
└── README.md
```

## Dataset
Source: Kaggle — Spotify Tracks Dataset
Link: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
