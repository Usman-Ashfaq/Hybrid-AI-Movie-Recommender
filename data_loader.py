import pandas as pd

def load_data():
    # Loading dataset
    df = pd.read_csv("tmdb_5000_movies.csv", low_memory=False)

    # Selecting useful columns
    df = df[['title', 'genres', 'release_date', 'runtime', 'vote_average', 'popularity']]

    # Removing missing values
    df.dropna(inplace=True)

    # Converting year
    df['year'] = df['release_date'].apply(lambda x: int(str(x)[:4]) if str(x) != 'nan' else 2000)

    # Removing invalid runtime
    df = df[df['runtime'] > 0]

    # Converting genres to string
    df['genres'] = df['genres'].astype(str)

    return df.to_dict(orient="records")