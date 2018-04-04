import numpy as np
import matplotlib.pyplot as plt

# generate sample data
x = np.linspace(0, 20, 1000)
y = np.sin(x)

fig = plt.figure(figsize=(8, 4))                        # get an empty figure and add an Axes
ax = fig.add_subplot(1, 1, 1)                           # row-col-num
ax.plot(x, y, 'b-', linewidth=2, label='Sample label')  # line plot data on the Axes
# add title, labels and legend, etc.
ax.set_ylabel('y axis label', fontsize=16)              # y label
ax.set_xlabel('x axis label', fontsize=16)              # x label
ax.legend(loc='best')
ax.grid(True)                                           # show grid
fig.suptitle("Sample Plot Title")                       # title
fig.tight_layout(pad=1)                                 # tidy layout
fig.savefig("filename.png", dpi=125)
plt.show()
