
import numpy as np
import matplotlib.pyplot as plt

# generate sample data
data = np.array([15, 8, 4])
labels = ['Feature Engineering', 'Model Tuning', 'Model Building']
explode = (0, 0.1, 0)   # explode feature engineering
colrs=['cyan', 'tan', 'wheat']

# plot
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.pie(data, explode=explode, labels=labels, autopct='%1.1f%%', startangle=270, colors=colrs)
ax.axis('equal')    # keep it a circle

# tidy-up and save
fig.suptitle("ML Pie")
# fig.savefig('filename.png', dpi=125)
plt.show()
