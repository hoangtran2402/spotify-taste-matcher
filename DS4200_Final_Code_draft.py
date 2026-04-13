import os
import pandas as pd
import numpy as np


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

FEATURE_WEIGHTS = {
    "popularity": 0.75,
    "duration_ms": 0.35,
    "danceability": 1.25,
    "energy": 1.35,
    "loudness": 0.50,
    "speechiness": 0.60,
    "acousticness": 1.10,
    "instrumentalness": 0.85,
    "liveness": 0.55,
    "valence": 1.15,
    "tempo": 1.00
}

GENRE_BONUS = 0.18


def load_data(filename="songs_normalize.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    df = pd.read_csv(file_path)

    df = df.rename(columns={
        "song": "track_name",
        "artist": "artists",
        "genre": "track_genre"
    })

    return df


def split_genres(genre_value):
    if pd.isna(genre_value):
        return []

    return [g.strip().lower() for g in str(genre_value).split(",") if g.strip()]


def clean_data(df):
    required_cols = [
        "track_name",
        "artists",
        "track_genre",
        "year",
        *FEATURE_COLUMNS
    ]

    df = df.dropna(subset=required_cols)
    df = df.drop_duplicates(subset=["track_name", "artists"], keep="first")

    df = df[
        (df["duration_ms"] >= 60000) &
        (df["duration_ms"] <= 420000)
    ]

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    df["genre_list"] = df["track_genre"].apply(split_genres)
    df = df[df["genre_list"].map(len) > 0]

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


def get_genre_options(df):
    unique_genres = set()

    for genres in df["genre_list"]:
        for genre in genres:
            unique_genres.add(genre)

    return sorted(unique_genres)


def print_intro():
    print("\nSpotify Taste Matcher")
    print("-" * 45)
    print("Answer each question from 0 to 10.")
    print("0 = not at all")
    print("10 = very much")
    print("Your answers will be converted to a 0.00 to 1.00 taste profile.\n")


def get_slider_input(question):
    while True:
        try:
            raw = input(f"{question} (0-10): ").strip()
            value = float(raw)

            if 0 <= value <= 10:
                return round(value / 10, 2)

            print("Please enter a number between 0 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 10.")


def get_user_feature_profile():
    print_intro()
    profile = {}

    for feature in FEATURE_COLUMNS:
        question = DISPLAY_QUESTIONS[feature]
        profile[f"{feature}_norm"] = get_slider_input(question)

    return profile


def get_year_range_input(df):
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    print("\nRelease year range")
    print("-" * 45)
    print(f"Oldest track year : {min_year}")
    print(f"Newest track year : {max_year}")
    print("Enter a year range like: 1960-2010")
    print("Press Enter for the full range.\n")

    while True:
        raw = input("Year range: ").strip()

        if raw == "":
            return min_year, max_year

        if "-" not in raw:
            print("Please use the format start-end, like 1960-2010.")
            continue

        left, right = raw.split("-", 1)
        left = left.strip()
        right = right.strip()

        try:
            start_year = int(left)
            end_year = int(right)
        except ValueError:
            print("Both values must be valid years.")
            continue

        if start_year > end_year:
            print("The start year must be less than or equal to the end year.")
            continue

        if end_year < min_year or start_year > max_year:
            print("That range is completely outside the dataset.")
            continue

        start_year = max(start_year, min_year)
        end_year = min(end_year, max_year)

        return start_year, end_year


def filter_by_year_range(df, year_range):
    start_year, end_year = year_range
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()


def display_genres(genres, per_line=4):
    print("\nAvailable genres:")
    print("-" * 45)

    for i, genre in enumerate(genres, start=1):
        print(f"{i:>3}. {genre:<20}", end="")
        if i % per_line == 0:
            print()

    if len(genres) % per_line != 0:
        print()

    print()


def get_user_genres(genres):
    display_genres(genres)

    print("Choose up to 3 genres by number, separated by commas.")
    print("Example: 2, 8, 15")
    print("Or press Enter to skip genre preference.\n")

    while True:
        raw = input("Your genre choices: ").strip()

        if raw == "":
            return []

        try:
            selected_indices = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
            selected_indices = list(dict.fromkeys(selected_indices))

            if len(selected_indices) > 3:
                print("Please choose no more than 3 genres.")
                continue

            if not all(1 <= idx <= len(genres) for idx in selected_indices):
                print("One or more genre numbers are out of range.")
                continue

            selected_genres = [genres[idx - 1] for idx in selected_indices]
            return selected_genres

        except ValueError:
            print("Invalid input. Please enter genre numbers separated by commas.")


def compute_match_score(row, user_profile, selected_genres=None,
                        feature_weights=None, genre_bonus=0.18):
    if feature_weights is None:
        feature_weights = FEATURE_WEIGHTS

    weighted_diff_sum = 0.0
    total_weight = 0.0

    for feature in FEATURE_COLUMNS:
        col = f"{feature}_norm"
        song_value = row[col]
        user_value = user_profile[col]
        diff = abs(song_value - user_value)

        weight = feature_weights.get(feature, 1.0)
        weighted_diff_sum += diff * weight
        total_weight += weight

    weighted_mean_diff = weighted_diff_sum / total_weight
    score = 1 - weighted_mean_diff

    if selected_genres:
        if any(genre in row["genre_list"] for genre in selected_genres):
            score += genre_bonus

    return round(score, 4)


def recommend_tracks(df, user_profile, selected_genres, top_n=5,
                     feature_weights=None, genre_bonus=0.18):
    df = df.copy()

    df["match_score"] = df.apply(
        lambda row: compute_match_score(
            row,
            user_profile,
            selected_genres,
            feature_weights=feature_weights,
            genre_bonus=genre_bonus
        ),
        axis=1
    )

    selected_rows = []

    if selected_genres:
        for genre in selected_genres:
            genre_df = df[df["genre_list"].apply(lambda genres: genre in genres)].sort_values(
                "match_score", ascending=False
            )

            if not genre_df.empty:
                selected_rows.append(genre_df.iloc[0])

    if selected_rows:
        guaranteed_df = pd.DataFrame(selected_rows)
        guaranteed_df = guaranteed_df.loc[~guaranteed_df.index.duplicated(keep="first")]
        used_indices = guaranteed_df.index.tolist()
    else:
        guaranteed_df = pd.DataFrame(columns=df.columns)
        used_indices = []

    remaining_needed = max(0, top_n - len(guaranteed_df))

    remaining_df = df.drop(index=used_indices).sort_values("match_score", ascending=False)
    filler_df = remaining_df.head(remaining_needed)

    final_df = pd.concat([guaranteed_df, filler_df], axis=0)
    final_df = final_df.sort_values("match_score", ascending=False).head(top_n)

    return final_df[
        [
            "track_name",
            "artists",
            "year",
            "track_genre",
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
    ]


def show_user_profile(user_profile, selected_genres, year_range):
    print("\nYour normalized taste profile")
    print("-" * 45)

    for feature in FEATURE_COLUMNS:
        value = user_profile[f"{feature}_norm"]
        print(f"{feature:<18}: {value:.2f}")

    print(f"\nSelected genres     : {selected_genres if selected_genres else 'None'}")
    print(f"Selected year range : {year_range[0]}-{year_range[1]}")


def show_weights():
    print("\nCurrent feature weights")
    print("-" * 45)
    for feature, weight in FEATURE_WEIGHTS.items():
        print(f"{feature:<18}: {weight:.2f}")
    print(f"\nGenre bonus         : {GENRE_BONUS:.2f}")


def show_recommendations(top_tracks):
    print("\nTop matching tracks")
    print("-" * 90)

    if top_tracks.empty:
        print("No tracks matched the selected filters.")
        return

    for i, (_, row) in enumerate(top_tracks.iterrows(), start=1):
        print(f"{i}. {row['track_name']}")
        print(f"   Artist         : {row['artists']}")
        print(f"   Year           : {row['year']}")
        print(f"   Genre          : {row['track_genre']}")
        print(f"   Match Score    : {row['match_score']:.4f}")
        print(f"   Popularity     : {row['popularity']}")
        print(f"   Duration (ms)  : {row['duration_ms']}")
        print(f"   Danceability   : {row['danceability']}")
        print(f"   Energy         : {row['energy']}")
        print(f"   Loudness       : {row['loudness']}")
        print(f"   Speechiness    : {row['speechiness']}")
        print(f"   Acousticness   : {row['acousticness']}")
        print(f"   Instrumental   : {row['instrumentalness']}")
        print(f"   Liveness       : {row['liveness']}")
        print(f"   Valence        : {row['valence']}")
        print(f"   Tempo          : {row['tempo']}")
        print("-" * 90)


def main():
    df = load_data("songs_normalize.csv")
    df = clean_data(df)
    df = normalize_features(df)

    show_weights()

    user_profile = get_user_feature_profile()
    year_range = get_year_range_input(df)

    df_filtered = filter_by_year_range(df, year_range)

    if df_filtered.empty:
        print("\nNo songs fall inside that year range.")
        return

    genres = get_genre_options(df_filtered)
    selected_genres = get_user_genres(genres)

    show_user_profile(user_profile, selected_genres, year_range)

    top_tracks = recommend_tracks(
        df=df_filtered,
        user_profile=user_profile,
        selected_genres=selected_genres,
        top_n=5,
        feature_weights=FEATURE_WEIGHTS,
        genre_bonus=GENRE_BONUS
    )

    show_recommendations(top_tracks)


if __name__ == "__main__":
    main()