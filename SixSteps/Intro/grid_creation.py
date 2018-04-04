
import numpy as np
import matplotlib.pyplot as plt

# Simple subplot grid layouts
fig = plt.figure(figsize=(8, 4))
fig.text(x=0.01, y=0.01, s='Figure', color='#888888', ha='left', va='bottom', fontsize=20)

for i in range(4):
    # fig.add subplot(nrows, ncols, num
    ax = fig.add_subplot(2, 2, i+1)
    ax.text(x=0.01, y=0.01, s = 'Subplot 2 2 '+str(i+1), color='red', ha='left', va='bottom', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('Subplots')
#fig.savefig('filename.png', dpi=125)
plt.show()
