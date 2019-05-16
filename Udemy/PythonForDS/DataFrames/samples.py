
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/P4-Demographic-Data.csv')
print(df.head())

print(df.describe().transpose())

print(df[(df['Birth rate'] > 40) & (df['Internet users'] < 2)])

# .at - for labels. Important: even integers are treated as labels
# .iat - for integer location
print(df.iat[3, 4])
print(df.at[2, 'Birth rate'])

vis1 = sns.distplot(df['Internet users'])
vis2 = sns.lmplot(x='Internet users', y='Birth rate', data=df)
vsi3 = sns.lmplot(data=df, x='Internet users', y='Birth rate', fit_reg=False, hue='Income Group', size=10)
vis4 = sns.lmplot(data=df, x='Internet users', y='Birth rate', fit_reg=False, hue='Income Group', size=8,
                  scatter_kws={"s": 100})

plt.show()
