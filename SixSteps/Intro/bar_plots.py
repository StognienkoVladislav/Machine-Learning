
import numpy as np
import matplotlib.pyplot as plt

# get the data
N = 4
labels = list('ABCD')
data = np.array(range(N)) + np.random.rand(N)

# plot the data
fig, ax = plt.subplots(figsize=(8, 3.5))
width = 0.5
tickLocations = np.arange(N)
rectLocations = tickLocations - (width/2.0)

# for color either HEX value of the name of the color can be used
ax.bar(rectLocations, data, width, color='lightblue', edgecolor='#1f10ed', linewidth=4.0)

# tidy-up the plot
ax.set_xticks(ticks=tickLocations)
ax.set_xticklabels(labels)
ax.set_xlim(min(tickLocations)-0.6, max(tickLocations)+0.6)
ax.set_yticks(range(N)[1:])
ax.set_ylim((0, N))
ax.yaxis.grid(True)
ax.set_ylabel('y axis label', fontsize=8)
ax.set_xlabel('x axis label', fontsize=8)

# title and save
fig.suptitle("Bar Plot")
fig.tight_layout(pad=2)
# fig.savefig("filename.png", dpi=125)
plt.show()
