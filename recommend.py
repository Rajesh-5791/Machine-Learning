import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

users = pd.read_csv("./dataset/u.data", sep='\t', names=["user_id", "movie_id", "rating", "time_stamp"])
movies = pd.read_csv(
    "./dataset/u.item", sep='|', encoding='latin-1', header=None,
    names=["movie_id", "movie_title", "release_date", "video_release_date", "IMDbURL", "unknown", "action", "adventure", "animation", 
           "children's", "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror", "musical", "mystery", 
           "romance", "sci-fi", "thriller", "war", "western"]
)

genre_columns = [
    'unknown', 'action', 'adventure', 'animation', "children's", 'comedy', 'crime', 'documentary',
    'drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western'
]

movies['genres'] = movies[genre_columns].apply(lambda x: [genre_columns[i] for i in range(len(x)) if x.iloc[i] == 1], axis=1)
users_and_movies = pd.merge(users, movies, on='movie_id', how='inner')
users_and_movies = users_and_movies[['user_id', 'movie_id', 'movie_title', 'genres']]
# print(users_and_movies.head(5))

user_genres = users_and_movies.explode('genres')
# print(user_genres)

user_genre_summary = user_genres.groupby('user_id')['genres'].apply(list).reset_index()
# print(user_genre_summary)

user_genre_summary['genres'] = user_genre_summary['genres'].apply(lambda x: list(set(x)))
# print(user_genre_summary['genres'])

movie_genre_matrix = movies[genre_columns]
# print(movie_genre_matrix)

user_genre_vectors = []
for _, row in user_genre_summary.iterrows():
    genre_vector = np.zeros(len(genre_columns))
    for genre in row['genres']:
        if genre in genre_columns:
            genre_vector[genre_columns.index(genre)] = 1
    user_genre_vectors.append(genre_vector)

user_genre_matrix = np.array(user_genre_vectors)
# print(user_genre_matrix)

similarity_scores = cosine_similarity(user_genre_matrix, movie_genre_matrix)

def recommend_movies(user_id, similarity_scores, users_and_movies):
    watched_movies = users_and_movies[users_and_movies['user_id'] == user_id]['movie_id'].tolist()
    
    if not watched_movies: 
        popular_movies = users_and_movies['movie_id'].value_counts().index[:10]
        return popular_movies
        
    user_similarity_scores = similarity_scores[user_id - 1]

    movie_similarities = list(zip(movies['movie_id'], user_similarity_scores))
    movie_similarities.sort(key=lambda x: x[1], reverse=True)

    recommended_movies = [movie for movie, _ in movie_similarities if movie not in watched_movies]    
    return recommended_movies[:10]

user_id = 20
recommended_movies = recommend_movies(user_id, similarity_scores, users_and_movies)
print(f"\nTop 10 recommended movies for user {user_id}:")
for idx, movie_id in enumerate(recommended_movies, 1):
    movie_title = movies[movies['movie_id'] == movie_id]['movie_title'].values[0]
    print(f"{idx}. {movie_title}")
