from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

FEATURE_COLUMNS = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

DISPLAY_QUESTIONS = {
    "popularity": "How much do you want popular / mainstream songs?",
    "duration_ms": "How much do you prefer longer songs?",
    "danceability": "How much do you like danceable songs?",
    "energy": "How much do you like high-energy songs?",
    "loudness": "How much do you like louder, more intense songs?",
    "speechiness": "How much do you like spoken-word / talk-heavy songs?",
    "acousticness": "How much do you like acoustic songs?",
    "instrumentalness": "How much do you like instrumental songs?",
    "liveness": "How much do you like live-performance sounding songs?",
    "valence": "How much do you like happier / more upbeat songs?",
    "tempo": "How much do you like faster-paced songs?"
}

_cached_df = None


def load_data(filename="songs_normalize.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


def clean_data(df):
    required_cols = ["song", "artist", "genre", *FEATURE_COLUMNS]

    df = df.drop_duplicates(subset=["song", "artist"], keep="first")
    df = df.dropna(subset=required_cols)

    return df


def normalize_features(df):
    df = df.copy()

    for col in FEATURE_COLUMNS:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_max == col_min:
            df[f"{col}_norm"] = 0.0
        else:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)

        df[f"{col}_norm"] = df[f"{col}_norm"].round(2)

    return df


def get_prepared_data():
    global _cached_df
    if _cached_df is None:
        df = load_data("songs_normalize.csv")
        df = clean_data(df)

        # Remove 2020 because it may not represent a full year
        if "year" in df.columns:
            df = df[df["year"] <= 2019]

        df = normalize_features(df)
        _cached_df = df
    return _cached_df


def get_genre_options(df):
    all_genres = set()

    for value in df["genre"].dropna():
        parts = [g.strip() for g in str(value).split(",")]
        for genre in parts:
            if genre:
                all_genres.add(genre)

    return sorted(all_genres)


def compute_match_score(row, user_profile, selected_genres=None, genre_bonus=0.08):
    diffs = []

    for feature in FEATURE_COLUMNS:
        col = f"{feature}_norm"
        song_value = row[col]
        user_value = user_profile[col]
        diffs.append(abs(song_value - user_value))

    mean_diff = np.mean(diffs)
    score = 1 - mean_diff

    if selected_genres:
        song_genres = [g.strip() for g in str(row["genre"]).split(",")]
        if any(genre in song_genres for genre in selected_genres):
            score += genre_bonus

    return round(score, 4)


def recommend_tracks(df, user_profile, selected_genres, top_n=5):
    df = df.copy()

    df["match_score"] = df.apply(
        lambda row: compute_match_score(row, user_profile, selected_genres),
        axis=1
    )

    selected_rows = []

    if selected_genres:
        for genre in selected_genres:
            genre_df = df[
                df["genre"].apply(
                    lambda x: genre in [g.strip() for g in str(x).split(",")]
                )
            ].sort_values("match_score", ascending=False)

            if not genre_df.empty:
                selected_rows.append(genre_df.iloc[0])

    if selected_rows:
        guaranteed_df = pd.DataFrame(selected_rows).drop_duplicates(subset=["song", "artist"])
        used_indices = guaranteed_df.index.tolist()
    else:
        guaranteed_df = pd.DataFrame(columns=df.columns)
        used_indices = []

    remaining_needed = top_n - len(guaranteed_df)

    remaining_df = df.drop(index=used_indices).sort_values(
        "match_score", ascending=False
    )

    filler_df = remaining_df.head(remaining_needed)

    final_df = pd.concat([guaranteed_df, filler_df], axis=0)
    final_df = final_df.sort_values("match_score", ascending=False).head(top_n)

    columns_to_show = [
        "song",
        "artist",
        "genre",
        "year",
        "explicit",
        "match_score",
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]

    return final_df[columns_to_show]


@app.route("/", methods=["GET", "POST"])
def index():
    df = get_prepared_data()
    genres = get_genre_options(df)
    recommendations = []
    current_values = {feature: 5 for feature in FEATURE_COLUMNS}
    selected_genres = []

    if request.method == "POST":
        user_profile = {}

        for feature in FEATURE_COLUMNS:
            raw_value = float(request.form.get(feature, 5))
            current_values[feature] = raw_value
            user_profile[f"{feature}_norm"] = round(raw_value / 10, 2)

        selected_genres = request.form.getlist("genres")[:3]

        top_tracks = recommend_tracks(
            df=df,
            user_profile=user_profile,
            selected_genres=selected_genres,
            top_n=5
        )

        recommendations = top_tracks.to_dict(orient="records")

    return render_template(
        "index.html",
        feature_columns=FEATURE_COLUMNS,
        display_questions=DISPLAY_QUESTIONS,
        genres=genres,
        recommendations=recommendations,
        current_values=current_values,
        selected_genres=selected_genres
    )


# ---------------------------------------------------------------------------
# Visualization data endpoint (DS4200 — added, does not modify existing code)
# ---------------------------------------------------------------------------

import json as _json

@app.route("/api/viz-data")
def viz_data():
    """Return pre-aggregated data needed by the five DS4200 visualizations."""
    df = get_prepared_data().copy()

    # Derive a single primary genre per track (first listed)
    df["primary_genre"] = df["genre"].apply(
        lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "Unknown"
    )

    # Chart 1 + Chart 3 song-level data
    songs_cols = [
        "song", "artist", "danceability", "energy", "popularity",
        "primary_genre", "year", "valence", "acousticness"
    ]
    songs_sample = df[songs_cols].sample(n=min(600, len(df)), random_state=42)
    songs_data = songs_sample.to_dict(orient="records")

    # Chart 1
    genre_pop = (
        df[["primary_genre", "popularity"]]
        .rename(columns={"primary_genre": "genre"})
        .to_dict(orient="records")
    )

    # Correlation heatmap
    numeric_cols = [
        "popularity",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]
    corr_matrix = df[numeric_cols].corr().round(3)
    corr_data = [
        {"row": r, "col": c, "value": float(corr_matrix.loc[r, c])}
        for r in numeric_cols
        for c in numeric_cols
    ]

    # Chart 4
    artist_year = (
        df.groupby(["year", "artist"])
        .size()
        .reset_index(name="count")
        .sort_values(["year", "count"], ascending=[True, False])
    )
    artist_year_data = artist_year.to_dict(orient="records")

    # Chart 5 - multiple selectable metrics
    metric_time = {}
    metric_columns = ["energy", "danceability", "valence", "acousticness"]

    for metric in metric_columns:
        metric_df = (
            df.groupby("year")[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: "value"})
            .round(4)
        )
        metric_time[metric] = metric_df.to_dict(orient="records")

    payload = {
        "songs": songs_data,
        "genre_popularity": genre_pop,
        "correlation": corr_data,
        "numeric_cols": numeric_cols,
        "artist_year": artist_year_data,
        "metric_time": metric_time,
        "years": sorted(df["year"].dropna().unique().astype(int).tolist()),
    }
    return app.response_class(
        response=_json.dumps(payload),
        mimetype="application/json"
    )


if __name__ == "__main__":
    app.run(debug=True)