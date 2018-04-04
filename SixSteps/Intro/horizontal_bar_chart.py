
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

ax.barh(rectLocations, data, width, color='lightblue')
# tidy-up the plot
ax.set_yticks(ticks=tickLocations)
ax.set_yticklabels(labels)
ax.set_ylim(min(tickLocations)-0.6, max(tickLocations)+0.6)
ax.xaxis.grid(True)
ax.set_ylabel('y axis label', fontsize=8)
ax.set_xlabel('x axis label', fontsize=8)

fig.suptitle("Bar plot")
fig.tight_layout(pad=2)
#fig.savefig('filename.png', dpi=125)
plt.show()
