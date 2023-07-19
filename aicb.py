import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

# Merge datasets
merged_df = pd.merge(movies_df, ratings_df)

# Pivot table to get user ratings for each movie
ratings_pivot = merged_df.pivot_table(index=['movieId'], columns=['userId'], values='rating')

# Fill missing values with 0
ratings_pivot = ratings_pivot.fillna(0)

# Calculate pairwise cosine similarity between movies
movie_similarity = cosine_similarity(ratings_pivot)

# Define chatbot function
def movie_recommender():
    print("Welcome to the Movie Recommender Chatbot!")
    while True:
        query = input("What movie are you in the mood for today? ")
        # Find movie ID in dataset
        movie_id = movies_df[movies_df['title'].str.contains(query, case=False)]['movieId'].values[0]
        # Get similarity scores for other movies
        similar_scores = movie_similarity[movie_id]
        # Sort movies by similarity score
        similar_movies = list(enumerate(similar_scores))
        sorted_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)
        # Print recommended movies
        print(f"We recommend '{movies_df[movies_df['movieId']==movie_id]['title'].values[0]}'!")
        print("Here are some similar movies you might also enjoy:")
        for i in range(1,6):
            movie_title = movies_df[movies_df['movieId']==sorted_movies[i][0]]['title'].values[0]
            print(f"- {movie_title}")
        print()

# Call chatbot function
movie_recommender()
