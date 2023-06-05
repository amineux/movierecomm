import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie ratings dataset
ratings = pd.read_csv('ratings.csv')

# Compute item-item similarity matrix
item_similarity = cosine_similarity(ratings.T)

# Function to get movie recommendations based on user input
def get_movie_recommendations(movie_title, top_n=5):
    movie_index = ratings.columns.get_loc(movie_title)
    similar_movies = list(enumerate(item_similarity[movie_index]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    top_similar_movies = [ratings.columns[i[0]] for i in similar_movies[1:top_n+1]]
    return top_similar_movies

# Get movie recommendations for a given movie
recommendations = get_movie_recommendations('Toy Story')

# Print the recommended movies
print(recommendations)

