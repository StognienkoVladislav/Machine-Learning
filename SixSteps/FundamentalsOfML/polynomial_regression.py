
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load data
df = pd.read_csv('../Data/data_3/Grade_Set_2.csv')
print(df)

# Simple scatter plot
df.plot(kind='scatter', x='Hours_Studied', y='Test_Grade', title='Grade vs Hours Studied')

# check the correlation between variables
print("Correlation matrix: ", df.corr())

# Create linear regression object
lr = lm.LinearRegression()

x = df.Hours_Studied[:, np.newaxis]
y = df.Test_Grade

# Train the model using the training sets
lr.fit(x, y)

# plotting fitted line
plt.scatter(x, y, color='black')
plt.plot(x, lr.predict(x), color='blue', linewidth=3)
plt.title("Grade vs Hours Studied")
plt.show()
print("R squared", r2_score(y, lr.predict(x)))


# r-squared for different polynomial degrees
lr = lm.LinearRegression()

x = df.Hours_Studied                # independent variable
y = df.Test_Grade                   # dependent variable

# NumPy`s vander function will return powers of the input vector

for deg in [1, 2, 3, 4, 5]:
    lr.fit(np.vander(x, deg+1), y)
    y_lr = lr.predict(np.vander(x, deg + 1))
    plt.plot(x, y_lr, label='degree ' + str(deg))
    plt.legend(loc=2)
    print(r2_score(y, y_lr))
plt.plot(x, y, 'ok')
plt.show()