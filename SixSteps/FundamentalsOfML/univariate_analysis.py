
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


iris = datasets.load_iris()

# Let`s convert to dataframe
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['species'])

# replace the values with class labels
iris.species = np.where(iris.species == 0.0, 'setosa', np.where(iris.species == 1.0, 'versicolor', 'virginica'))

# let`s remove spaces from column name
iris.columns = iris.columns.str.replace(' ', '')
# print(iris.describe())

# print(iris['species'].value_counts())

# Set the size of plot
fig = plt.figure(figsize=(15, 8))

iris.hist()             # plot histogram
plt.suptitle("Histogram", fontsize=16)
plt.show()

iris.boxplot()          # plot boxplot
plt.title('Bar Plot', fontsize=16)
plt.show()

# print the mean for each column by species
iris.groupby(by='species').mean()

# plot for mean of each feature for each label class
iris.groupby(by='species').mean().plot(kind='bar')

plt.title('Class vs Measurements')
plt.ylabel('mean measurement(cm)')
plt.xticks(rotation=0)  # manage the xticks rotation
plt.grid(True)
# Use bbox_to_anchor option to place the legend outside plot area to be tidy
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# Correlation matrix
corr = iris.corr()
print(corr)

import statsmodels.api as sm
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()
