"""
Spotify Top Songs Analysis
Author: Rafael Cruz-Camacho
Dataset: Spotify Tracks Dataset (Kaggle - maharshipandya)

Run from project root in VS Code terminal:
    python notebooks/analysis.py

Output:
    outputs/  --> PNG chart files
    data/clean/spotify_clean.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────
RAW       = r"C:\Users\rafcr\Documents\projects\spotify-analysis-python\data\raw\dataset.csv"
CLEAN_DIR = r"C:\Users\rafcr\Documents\projects\spotify-analysis-python\data\clean"
OUT_DIR   = r"C:\Users\rafcr\Documents\projects\spotify-analysis-python\outputs"
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

# ── Style ─────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="husl")
SPOTIFY_GREEN = "#1DB954"
plt.rcParams.update({'figure.dpi': 150, 'font.family': 'sans-serif'})

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv(RAW)
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
print(f"Loaded: {df.shape[0]:,} tracks x {df.shape[1]} columns\n")

# ── Clean ─────────────────────────────────────────────────────
df = df.dropna(subset=['popularity', 'track_name', 'artists'])
df = df.drop_duplicates(subset=['track_id'])
df['duration_min'] = df['duration_ms'] / 60000
df['explicit'] = df['explicit'].astype(bool)

# Map key numbers to note names
key_map = {0:'C',1:'C#',2:'D',3:'D#',4:'E',5:'F',
           6:'F#',7:'G',8:'G#',9:'A',10:'A#',11:'B'}
df['key_name'] = df['key'].map(key_map)

print(f"Unique genres  : {df['track_genre'].nunique()}")
print(f"Unique artists : {df['artists'].nunique():,}")
print(f"Popularity range: {df['popularity'].min()} - {df['popularity'].max()}\n")

# ── Analysis 1: Top 15 Genres by Average Popularity ──────────
print("Building Chart 1: Top genres by popularity...")
genre_pop = (
    df.groupby('track_genre')['popularity']
    .agg(avg_popularity='mean', track_count='count')
    .query('track_count >= 50')
    .sort_values('avg_popularity', ascending=False)
    .head(15)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(genre_pop['track_genre'], genre_pop['avg_popularity'],
               color=SPOTIFY_GREEN, alpha=0.85)
ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
ax.set_xlabel('Average Popularity Score (0-100)')
ax.set_title('Top 15 Genres by Average Popularity', fontsize=14, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.set_xlim(0, genre_pop['avg_popularity'].max() * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_top_genres_popularity.png'))
plt.close()
print("  Saved: 01_top_genres_popularity.png")

# ── Analysis 2: What makes a song popular? Correlation heatmap
print("Building Chart 2: Feature correlations with popularity...")
feature_cols = ['popularity','danceability','energy','loudness',
                'speechiness','acousticness','instrumentalness',
                'liveness','valence','tempo','duration_min']

corr = df[feature_cols].corr()[['popularity']].drop('popularity').round(3)
corr = corr.sort_values('popularity', ascending=True)

fig, ax = plt.subplots(figsize=(7, 7))
colors = [SPOTIFY_GREEN if x > 0 else '#E53935' for x in corr['popularity']]
bars = ax.barh(corr.index, corr['popularity'], color=colors, alpha=0.85)
ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xlabel('Correlation with Popularity')
ax.set_title('Audio Features vs Song Popularity\n(Correlation Coefficients)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(-0.45, 0.45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_feature_correlations.png'))
plt.close()
print("  Saved: 02_feature_correlations.png")

# ── Analysis 3: Energy vs Danceability scatter (top genres) ──
print("Building Chart 3: Energy vs danceability scatter...")
top_genres = df['track_genre'].value_counts().head(6).index
df_scatter = df[df['track_genre'].isin(top_genres)].sample(3000, random_state=42)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    df_scatter['energy'], df_scatter['danceability'],
    c=df_scatter['popularity'], cmap='YlGn',
    alpha=0.5, s=15
)
plt.colorbar(scatter, ax=ax, label='Popularity Score')
ax.set_xlabel('Energy')
ax.set_ylabel('Danceability')
ax.set_title('Energy vs Danceability\n(Color = Popularity, sample of 3,000 tracks)',
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_energy_vs_danceability.png'))
plt.close()
print("  Saved: 03_energy_vs_danceability.png")

# ── Analysis 4: Popularity distribution by explicit content ──
print("Building Chart 4: Explicit vs non-explicit popularity...")
fig, ax = plt.subplots(figsize=(9, 5))
for explicit, label, color in [(False, 'Clean', '#1DB954'), (True, 'Explicit', '#E53935')]:
    subset = df[df['explicit'] == explicit]['popularity']
    ax.hist(subset, bins=40, alpha=0.6, label=f"{label} (n={len(subset):,})", color=color)
ax.set_xlabel('Popularity Score')
ax.set_ylabel('Number of Tracks')
ax.set_title('Popularity Distribution: Explicit vs Clean Tracks',
             fontsize=13, fontweight='bold', pad=15)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_explicit_vs_clean.png'))
plt.close()
print("  Saved: 04_explicit_vs_clean.png")

# ── Analysis 5: Top 10 most popular tracks ───────────────────
print("\n=== Top 10 Most Popular Tracks ===")
top10 = (df.nlargest(10, 'popularity')
           [['track_name','artists','track_genre','popularity',
             'danceability','energy','tempo']]
           .reset_index(drop=True))
top10.index += 1
print(top10.to_string())

# ── Analysis 6: Key findings summary ─────────────────────────
print("\n=== KEY FINDINGS ===")
print(f"Most popular genre    : {genre_pop.iloc[0]['track_genre']} ({genre_pop.iloc[0]['avg_popularity']:.1f} avg)")

best_feature = corr['popularity'].abs().idxmax()
best_corr    = corr.loc[best_feature, 'popularity']
print(f"Strongest feature corr: {best_feature} (r={best_corr:.3f})")

explicit_avg = df.groupby('explicit')['popularity'].mean().round(1)
print(f"Avg popularity — Clean : {explicit_avg[False]}")
print(f"Avg popularity — Explicit: {explicit_avg[True]}")

print(f"\nTotal tracks analyzed : {len(df):,}")
print(f"Genres covered        : {df['track_genre'].nunique()}")

# ── Export clean CSV ──────────────────────────────────────────
clean_cols = ['track_name','artists','track_genre','popularity',
              'danceability','energy','loudness','acousticness',
              'instrumentalness','liveness','valence','tempo',
              'duration_min','explicit','key_name','mode']
df[clean_cols].to_csv(os.path.join(CLEAN_DIR, 'spotify_clean.csv'), index=False)
print(f"\nClean CSV --> {CLEAN_DIR}\\spotify_clean.csv")
print(f"Charts    --> {OUT_DIR}")
print("\nDone!")
