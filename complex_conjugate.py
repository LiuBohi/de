import matplotlib.pyplot as plt
import numpy as np

def plot_complex(z, axes=None):
	"""Input: number pair in the form of (x, y)
	   Output: return the axes and the plotting
	"""

    if axes is None:
    	fig, axes = plt.subplots(figsize=(8, 6))
    axes.scatter(z[0], z[1], color='orange', marker='o')
    axes.annotate('', xy=(z[0], z[1]),
                  xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->',
                                  lw=1, ls='--'))
    axes.text(z[0]+0.1, z[1]+0.1,
    	      f'({z[0]:.1f}, {z[1]:.1f})')

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_position(('data', 0))
    axes.set_xlabel("Re", loc='right', fontsize=14, fontweight='semibold')
    axes.spines['left'].set_position(('data', 0))
    axes.set_ylabel('Im', loc='top', rotation=0,labelpad=-30,
                    fontsize=14, fontweight='semibold')
    axes.set_xlim(-1.1*np.abs(z[0]), 1.1*np.abs(z[0]))
    axes.set_ylim(-1.1*np.abs(z[1]), 1.1*np.abs(z[1]))
    axes.grid(True, lw=0.2, color='grey', alpha=0.8)

    return axes


if __name__ == '__main__':
	z = (3, 4)  # This number pair can be changed.
	axes = plot_complex(z)
	z_conjugate = (z[0], -z[1])  # Make the conjugate of z
	plot_complex(z_conjugate, axes)
	axes.plot([z[0]]*10, np.linspace(z_conjugate[1], z, 10), 
		    linestyle='--', color='lightblue')
	axes.set_title(f'Compex number ({z[0]}, {z[1]}) and its conjugate',
		           x=0.5, y=-0.1, fontweight='semibold')
	# comment out the following line and change the path if you want to save plot.
	# plt.savefig(f'img/Compex number ({z[0]}, {z[1]}) and its conjugate.jpg', dpi=300)
	plt.show()

