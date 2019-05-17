
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


# Jointplots

j = sns.jointplot(data=movies, x='CriticRating', y='AudienceRating', kind='hex')
plt.show()


# Histograms

m1 = sns.distplot(movies.AudienceRating, bins=15)
plt.show()

# Stacked Histograms

genre_list = []
labels_list = []

for gen in movies.Genre.cat.categories:
    genre_list.append(movies[movies.Genre == gen].BudgetMillions)
    labels_list.append(gen)

hist = plt.hist(genre_list, bins=30, stacked=True, rwidth=1, label=labels_list)
plt.legend()
plt.show()

# KDE plot

k1 = sns.kdeplot(movies.CriticRating, movies.AudienceRating,
                 shade=True, shade_lowest=False, cmap='Reds')

# TIP:
k1_back = sns.kdeplot(movies.CriticRating, movies.AudienceRating,
                      cmap='Reds')
plt.show()


# Sub plots
budget_audience = sns.kdeplot(movies.BudgetMillions, movies.AudienceRating)
plt.show()


# 1 row, 2 columns
f, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
sub_k1 = sns.kdeplot(movies.BudgetMillions, movies.AudienceRating, ax=axes[0])
sub_k2 = sns.kdeplot(movies.BudgetMillions, movies.CriticRating, ax=axes[1])
sub_k1.set(xlim=(-20, 160))
plt.show()


# Violin and Box Plots
f2, axes_2 = plt.subplots(1, 2, figsize=(12, 6))
viol = sns.violinplot(data=movies, x='Genre', y='CriticRating', ax=axes_2[0])
box = sns.boxplot(data=movies, x='Genre', y='CriticRating', ax=axes_2[1])
plt.show()


# Facet grid
f_grid = sns.FacetGrid(movies, row='Genre', col='Year', hue='Genre')
kws = {
    's': 50,
    'linewidth': 0.5,
    'edgecolor': 'black'
}
f_grid = f_grid.map(plt.scatter, 'CriticRating', 'AudienceRating', **kws)
f_grid.set(xlim=(0, 100), ylim=(0, 100))

for ax in f_grid.axes.flat:
    ax.plot((0, 100), (0, 100), c='gray', ls='--')

f_grid.add_legend()
plt.show()


# Dash Board

sns.set_style('dark', {"axes.facecolor": "black"})
dash_f, dash_axes = plt.subplots(2, 2, figsize=(15, 15))

dash_1 = sns.kdeplot(movies.BudgetMillions, movies.AudienceRating, ax=dash_axes[0, 0],
                     shade=True, shade_lowest=True, cmap='inferno')
dash_1_back = sns.kdeplot(movies.BudgetMillions, movies.AudienceRating, ax=dash_axes[0, 0], cmap='cool')

dash_2 = sns.kdeplot(movies.BudgetMillions, movies.CriticRating, ax=dash_axes[0, 1],
                     shade=True, shade_lowest=True, cmap='inferno')
dash_2_back = sns.kdeplot(movies.BudgetMillions, movies.CriticRating, ax=dash_axes[0, 1], cmap='cool')

dash_3 = sns.violinplot(data=movies, x='Year', y='BudgetMillions', ax=dash_axes[1, 0], palette='cool')

dash_4 = sns.kdeplot(movies.CriticRating, movies.AudienceRating, shade=True, shade_lowest=False,
                     cmap='Blues_r', ax=dash_axes[1, 1])
dash_4_back = sns.kdeplot(movies.CriticRating, movies.AudienceRating, cmap='gist_gray_r', ax=dash_axes[1, 1])

dash_1.set(xlim=(-20, 160))
dash_2.set(xlim=(-20, 160))
plt.show()
