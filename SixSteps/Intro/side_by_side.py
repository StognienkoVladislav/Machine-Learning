
import numpy as np
import matplotlib.pyplot as plt

# generate sample data
pre = np.array([19, 6, 11, 9])
post = np.array([15, 11, 9, 8])
labels = ['Survey ' + x for x in list('ABCD')]

# the plot - left then right
fig, ax = plt.subplots(figsize=(8, 3.5))
width = 0.4
xlocs = np.arange(len(pre))
ax.bar(xlocs-width, pre, width, color='green', label='True')
ax.bar(xlocs, post, width, color='#1f10ed', label='False')
# labels, grids and title, then save
ax.set_xticks(ticks=range(len(pre)))
ax.set_xticklabels(labels)
ax.yaxis.grid(True)
ax.legend(loc='best')
ax.set_ylabel('Count')
fig.suptitle('Sample Chart')
fig.tight_layout(pad=1)
#fig.savefig('filename.png', dpi=125)
plt.show()
