# Tableau Guide

Use the CSV outputs under data/processed/tableau.

## Recommended sheets

1. Top movies by Bayesian score
- Source: top_movies_bayesian.csv
- Dimensions: movie_title
- Measures: bayesian_score, rating_count, mean_rating

2. Genre popularity
- Source: genre_popularity.csv
- Dimensions: genre
- Measures: rating_count, user_count, mean_rating, median_rating

3. Rating distribution
- Source: rating_distribution.csv
- Dimensions: rating
- Measure: count
