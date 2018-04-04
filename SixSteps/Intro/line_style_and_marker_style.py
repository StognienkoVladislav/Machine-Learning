
import numpy as np
import matplotlib.pyplot as plt

# get the Figure and Axes all at one
fig, ax = plt.subplots(figsize=(8, 4))
# plot some lines
N = 3 # the number of lines we will plot
styles = ['-', '--', ':', '-.']
markers = list('+ox')
x = np.linspace(0, 100, 20)

for i in range(N):          # add line-by-line
    y = x + x/5*i + i
    s = styles[i % len(styles)]
    m = markers[i % len(markers)]
    ax.plot(x, y, alpha=1, label='Line'+str(i+1)+' '+s+m, marker=m, linewidth=2, linestyle=s)

# add grid, legend, title and save
ax.grid(True)
ax.legend(loc='best', prop={'size': 'large'})
fig.suptitle('A Simple Line Plot')
#fig.savefig('filename.png', dpi=125)
plt.show()
