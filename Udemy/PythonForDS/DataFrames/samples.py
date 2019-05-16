
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/P4-Demographic-Data.csv')
print(df.head())

print(df.describe().transpose())
