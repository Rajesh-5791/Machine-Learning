import pandas as pd

# Reading 'u.data' and 'u.item' datasets
user_ratings = pd.read_csv("u.data", sep='\t', names=["user_id", "movie_id", "rating", "time_stamp"])
movie_info = pd.read_csv(
    "u.item", sep='|', encoding='latin-1', header=None,
    names=["movie_id", "movie_title", "release_date", "video_release_date", "IMDbURL", "unknown", "action", "adventure", "animation", 
           "children's", "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror", "musical", "mystery", 
           "romance", "sci-fi", "thriller", "war", "western"]
)

# Merging user_ratings and movie_info datasets on 'movie_id' column
user_ratings_with_movie_info = pd.merge(user_ratings, movie_info, on="movie_id")

# Preparing final custom dataset with user_id, movie_id and all genre columns and movie_title
final_custom_dataset = user_ratings_with_movie_info[["user_id", "movie_id", "movie_title", "action", "adventure", "animation", 
           "children's", "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror", "musical", "mystery", 
           "romance", "sci-fi", "thriller", "war", "western"]]

# Placing final customized dataset in 'custom_dataset.csv'
final_custom_dataset.to_csv("custom_dataset.csv", index=False)
final_dataset.to_csv("final_dataset.csv")