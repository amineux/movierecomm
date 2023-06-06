import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load movie data
movies = pd.read_csv('movies.csv')

# Preprocess movie data
movies['genres'] = movies['genres'].str.replace('|', ' ')
movies['overview'] = movies['overview'].fillna('')

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_movie_recommendations(title, cosine_sim=cosine_sim, movies=movies):
    # Find index of the movie
    idx = movies[movies['title'] == title].index[0]
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 similar movies
    top_movies = sim_scores[1:6]
    
    # Get movie titles
    movie_indices = [i[0] for i in top_movies]
    recommendations = movies['title'].iloc[movie_indices].values
    
    return recommendations

# Test the recommendation system
recommended_movies = get_movie_recommendations('Toy Story')
print(recommended_movies)

