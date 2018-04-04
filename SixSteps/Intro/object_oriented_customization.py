import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 4))

# Iterating the Axes within a Figure
for ax in fig.get_axes():
    pass # do something

plt.show()