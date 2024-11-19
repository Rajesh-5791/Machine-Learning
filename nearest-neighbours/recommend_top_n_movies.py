import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("custom_dataset.csv")
user_item_matrix = dataset.pivot_table(index='user_id', columns='movie_id', aggfunc='size', fill_value=0)
trainset, testset = train_test_split(user_item_matrix, test_size=0.3, random_state=1)
model = NearestNeighbors(metric='cosine')
model.fit(trainset)

def get_top_n_recommendations_for_existing_user(user_id, number_of_recommendations=10):    
    user_index = trainset.index.get_loc(user_id)
    distances, indices = model.kneighbors(trainset.iloc[user_index, :].values.reshape(1, -1), n_neighbors=number_of_recommendations + 1)
    similar_users = indices.flatten()[1:]
    similar_users_movies = trainset.iloc[similar_users].sum(axis=0).sort_values(ascending=False)
    watched_movies = set(trainset.columns[trainset.loc[user_id] > 0])
    recommended_movies = [movie for movie in similar_users_movies.index if movie not in watched_movies]
    top_movie_titles = dataset[dataset['movie_id'].isin(recommended_movies[:number_of_recommendations])]['movie_title'].unique().tolist()
    return top_movie_titles

def get_top_n_recommendations_for_new_user(number_of_recommendations=10):
    popular_movies = dataset['movie_id'].value_counts().index[:number_of_recommendations]
    top_movie_titles = dataset[dataset['movie_id'].isin(popular_movies)]['movie_title'].unique().tolist()
    return top_movie_titles

user_id = int(input("Enter user ID: "))
number_of_recommendations = 10

if user_id not in trainset.index:
    user_type = "new"
    top_movies = get_top_n_recommendations_for_new_user(number_of_recommendations)
else:
    user_type = "existing"
    top_movies = get_top_n_recommendations_for_existing_user(user_id, number_of_recommendations)

print(f"\nTop {number_of_recommendations} recommendations for {user_type} user (User ID: {user_id}):")
for index, movie in enumerate(top_movies, start=1):
    print(f"{index}. {movie}")
