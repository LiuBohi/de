import numpy as np
import matplotlib.pyplot as plt


def root_exp(x, h, ax):
    #TODO
    y1 = np.exp(2*x)
    y2 = (x*h - 1) / (1 + x*h)
    ax.plot(x, y1, label='LHS', color='red')
    ax.plot(x, y2, label='RHS', color='blue')
    ax.annotate(r'$y=e^{2x}$',
                xy=(x[-1], y1[-1]),
                xytext=(-50,-10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->',
                connectionstyle='arc3, rad=0'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(r'$\lambda < 0$'+f', h={h}')


def root_cot(x, h, ax):
    #TODO
    d = 0.01*np.pi
    X = [np.linspace((0+k)*np.pi+d, (1+k)*np.pi-d) for k in range(0,4)]
    X.insert(0, np.linspace(d, 0.5*np.pi-d))
    for i in range(len(X)):
        x1 = X[i]
        y1 = h*x1
        y2 = 1/np.tan(x1)
        ax.plot(x1, y1, label='LHS', color='red')
        ax.plot(x1, y2, label='RHS', color='blue')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))
    ax.set_xlabel(r'$x$')
    ax.set_xlim(0, 4*np.pi)
    ax.set_ylabel(r'$y$')
    ax.set_title(r'$\lambda>0$'+f', h={h}')
    ax.annotate(r'$y=\cot(x)$',
                xy=(X[-1][-1], 1/np.tan(X[-1][-1])),
                xytext=(-50,-10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->',
                connectionstyle='arc3, rad=0'))




if __name__ == "__main__":
    h = 1
    x = np.linspace(1e-1, 1, 50)
    fig, axes = plt.subplots(1, 2, figsize=(8,6))
    root_exp(x, h, axes[0])
    root_cot(x, h, axes[1])
    plt.savefig('img/h_1.jpg', dpi=300)
    plt.show()

    h = 2
    fig, axes = plt.subplots(1, 2, figsize=(8,6))
    root_exp(x, h, axes[0])
    root_cot(x, h, axes[1])
    plt.savefig('img/h_2.jpg', dpi=300)
    plt.show()

    h = -1
    fig, axes = plt.subplots(1, 2, figsize=(8,6))
    root_exp(x, h, axes[0])
    root_cot(x, h, axes[1])
    plt.savefig('img/h_-1.jpg', dpi=300)
    plt.show()
