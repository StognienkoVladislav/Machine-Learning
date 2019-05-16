
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


movies = pd.read_csv('data/P4-Movie-Ratings.csv')
print(movies.head())
print(len(movies))

movies.columns = ['Film', 'Genre', 'CriticRating', 'AudienceRating',
                  'BudgetMillions', 'Year']
print(movies.head())

movies.Film = movies.Film.astype('category')
movies.Genre = movies.Genre.astype('category')
movies.Year = movies.Year.astype('category')

print(movies.describe())
