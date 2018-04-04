
import pandas as pd

df = pd.DataFrame({'A': ['high', 'medium', 'low'],
                   'B' : [10, 20, 30]},
                  index=[0, 1, 2])
print(df)

df_with_dummies = pd.get_dummies(df, prefix='A', columns=['A'])
print(df_with_dummies)