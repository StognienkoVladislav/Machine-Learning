
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# get configuration file location
print(matplotlib.matplotlib_fname())

# get configuration current settings
print(matplotlib.rcParams)

# Change the default settings
plt.rc('figure', figsize=(8, 4), dpi=125, facecolor='white', edgecolor='white')
plt.rc('axes', facecolor='#e5e5e5', grid=True, linewidth=1.0, axisbelow=True)
plt.rc('grid', color='white', linestyle='-', linewidth=2.0, alpha=1.0)
plt.rc('xtick', direction='out')
plt.rc('ytick', direction='out')
plt.rc('legend', loc='best')
plt.show()
