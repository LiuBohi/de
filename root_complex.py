import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Visualize the way to find the root of the xomplex number


def plot_complex(z, axes=None):
    """
    """
    # TODO:
    if axes is None:
        fig, axes = plt.subplots(figsize=(8,6))
        axes.scatter(z[0], z[1], color='lightblue', marker='o')
    else:
        axes.scatter(z[0], z[1], color='lightblue', marker='o')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_position(('data', 0))
    axes.set_xlabel("Re", loc='right', fontsize=14, fontweight='semibold')
    axes.spines['left'].set_position(('data', 0))
    axes.set_ylabel('Im', loc='top', rotation=0,labelpad=-30,
                    fontsize=14, fontweight='semibold')
    axes.set_xlim(-1.1*z[0], 1.1*z[0])
    axes.set_ylim(-1.1*z[1], 1.1*z[1])
    axes.grid(True, lw=0.2, color='grey', alpha=0.8)
    # plt.savefig(f'{file_name}.jpg', dpi=300)
    return axes


def plot_complex_addition(z1, z2, axes=None):
    """
    """
    # TODO:
    z3 = (z1[0]+z2[0], z1[1]+z2[1])
    if axes is None:
        fig, axes = plt.subplots(figsize=(8,6))
    axes.scatter(z1[0], z1[1], color='lightblue', marker='o')
    axes.text(1.05*z1[0], z1[1],f'$z_1$=$({z1[0]:.1f}, {z1[1]:.1f})$')
    axes.scatter(z2[0], z2[1], color='lightblue', marker='o')
    axes.text(1.05*z2[0], z2[1],f'$z_2$=$({z2[0]:.1f}, {z2[1]:.1f})$')
    axes.scatter(z3[0], z3[1], color='orange', marker='o')
    axes.text(1.05*z3[0], z3[1],f'$z_3$=$z_1+z_2$\n   =$({z3[0]:.1f}, {z3[1]:.1f})$')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_position(('data', 0))
    axes.set_xlabel("Re", loc='right', fontsize=14, fontweight='semibold')
    axes.spines['left'].set_position(('data', 0))
    axes.set_ylabel('Im', loc='top', rotation=0,labelpad=-30,
                    fontsize=14, fontweight='semibold')
    axes.set_xlim(-1.1*np.max([z1[0],z2[0], z3[0]]),
                   1.1*np.max([z1[0], z2[0], z3[0]]))
    axes.set_ylim(-1.1*np.max([z1[1],z2[1], z3[1]]),
                   1.1*np.max([z1[1],z2[1], z3[1]]))
    axes.grid(True, lw=0.2, color='grey', alpha=0.8)
    # plt.savefig(f'{file_name}.jpg', dpi=300)
    return axes


def plot_complex_substraction(z1, z2, axes=None):
    """
    """
    # TODO:
    z3 = (z1[0]-z2[0], z1[1]-z2[1])
    if axes is None:
        fig, axes = plt.subplots(figsize=(8,6))

    axes.scatter(z1[0], z1[1], color='lightblue', marker='o')
    axes.text(1.05*z1[0], z1[1],f'$z_1$=$({z1[0]:.1f}, {z1[1]:.1f})$')
    axes.scatter(z2[0], z2[1], color='lightblue', marker='o')
    axes.text(1.05*z2[0], z2[1],f'$z_2$=$({z2[0]:.1f}, {z2[1]:.1f})$')
    axes.scatter(z3[0], z3[1], color='orange', marker='o')
    axes.text(1.05*z3[0], z3[1],f'$z_3$=$z_1-z_2$\n   =$({z3[0]:.1f}, {z3[1]:.1f})$')

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_position(('data', 0))
    axes.set_xlabel("Re", loc='right', fontsize=14, fontweight='semibold')
    axes.spines['left'].set_position(('data', 0))
    axes.set_ylabel('Im', loc='top', rotation=0,labelpad=-30,
                    fontsize=14, fontweight='semibold')
    axes.set_xlim(-1.1*np.max([z1[0],z2[0], z3[0]]),
                   1.1*np.max([z1[0], z2[0], z3[0]]))
    axes.set_ylim(-1.1*np.max([z1[1],z2[1], z3[1]]),
                   1.1*np.max([z1[1],z2[1], z3[1]]))
    axes.grid(True, lw=0.2, color='grey', alpha=0.8)
    # plt.savefig(f'{file_name}.jpg', dpi=300)
    return axes

def plot_complex_multiply(z1, z2, axes=None):
    """
    """
    # TODO:
    r1 = np.sqrt(z1[0]**2 + z1[1]**2)
    r2 = np.sqrt(z2[0]**2 + z2[1]**2)
    r3 = r1 / r2

    alpha = np.arctan(z1[1]/z1[0])
    if z1[0] < 0:
        alpha = alpha + np.pi
    a1 = np.linspace(0, alpha, 50)

    beta = np.arctan(z2[1]/z2[0])
    if z2[0] < 0:
        beta = beta + np.pi
    a2 = np.linspace(0, beta, 50)

    theta = alpha + beta
    a3 = np.linspace(0, theta, 50)

    if axes is None:
        fig, axes = plt.subplots(figsize=(8,6))
    axes.set_aspect('equal')
    plot_complex(z1, axes)
    #axes.text(1.1*z1[0], z1[1],
    #          r'$r_1$'+f'=${r1:.1f}, '+ r'\alpha' +f'={alpha:.1f}$')
    plot_complex(z2, axes)
    #axes.text(4.5*z2[0], z2[1],
    #          r'$r_2$'+f'=${r2:.1f}, '+ r'\beta' +f'={alpha:.1f}$')
    axes.scatter(r3*np.cos(theta), r3*np.sin(theta),
                 color='orange', marker='o')
    axes.annotate('', xy=(z1[0], z1[1]),
                  xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->',
                                  lw=1))
    axes.plot(r1*np.cos(a1)/2, r1*np.sin(a1)/2, color='black',
              ls='--', lw=1)
    axes.text(r1*np.cos(alpha/2)/2, r1*np.sin(alpha/2)/2,
              r'$\alpha$', fontweight='semibold')

    axes.annotate('', xy=(z2[0], z2[1]),
                  xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->'))
    axes.plot(r2*np.cos(a2)/2, r2*np.sin(a2)/2, color='black',
              ls='--', lw=1)
    axes.text(r2*np.cos(beta/2)/2, r2*np.sin(beta/2)/2,
              r'$\beta$',fontweight='semibold')

    axes.annotate('', xy=(r3*np.cos(theta), r3*np.sin(theta)),
                  xytext=(0,0),
                  arrowprops=dict(arrowstyle='->'))
    axes.plot(r3*np.cos(a3)/2, r3*np.sin(a3)/2, color='black',
              ls='--', lw=1)
    axes.text(r3*np.cos(theta/2)/2, 1.1*r3*np.sin(theta/2)/2,
              r'$\theta$',fontweight='semibold')


    xlim = np.max([z1[0], z2[0], np.abs(r3*np.cos(theta))])
    ylim = np.max([[z1[1], z2[1], np.abs(r3*np.sin(theta))]])
    if xlim > ylim:
        ylim = xlim
    else:
        xlim = ylim

    axes.set_xlim(-1.1*xlim, 1.1*xlim)
    axes.set_ylim(-1.1*ylim, 1.1*ylim)
    # plt.savefig(f'{file_name}.jpg', dpi=300)


def plot_complex_division(z1, z2, axes=None):
    """
    """
    # TODO:
    r1 = np.sqrt(z1[0]**2 + z1[1]**2)
    r2 = np.sqrt(z2[0]**2 + z2[1]**2)
    r3 = r1 * r2

    alpha = np.arctan(z1[1]/z1[0])
    if z1[0] < 0:
        alpha = alpha + np.pi
    a1 = np.linspace(0, alpha, 50)

    beta = np.arctan(z2[1]/z2[0])
    if z2[0] < 0:
        beta = beta + np.pi
    a2 = np.linspace(0, beta, 50)

    theta = alpha - beta
    a3 = np.linspace(0, theta, 50)

    if axes is None:
        fig, axes = plt.subplots(figsize=(8,6))
    axes.set_aspect('equal')
    plot_complex(z1, axes)
    #axes.text(1.1*z1[0], z1[1],
    #          r'$r_1$'+f'=${r1:.1f}, '+ r'\alpha' +f'={alpha:.1f}$')
    plot_complex(z2, axes)
    #axes.text(4.5*z2[0], z2[1],
    #          r'$r_2$'+f'=${r2:.1f}, '+ r'\beta' +f'={alpha:.1f}$')
    axes.scatter(r3*np.cos(theta), r3*np.sin(theta),
                 color='orange', marker='o')
    axes.annotate('', xy=(z1[0], z1[1]),
                  xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->',
                                  lw=1))
    axes.plot(r1*np.cos(a1)/2, r1*np.sin(a1)/2, color='black',
              ls='--', lw=1)
    axes.text(r1*np.cos(alpha/2)/2, r1*np.sin(alpha/2)/2,
              r'$\alpha$', fontweight='semibold')

    axes.annotate('', xy=(z2[0], z2[1]),
                  xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->'))
    axes.plot(r2*np.cos(a2)/2, r2*np.sin(a2)/2, color='black',
              ls='--', lw=1)
    axes.text(r2*np.cos(beta/2)/2, r2*np.sin(beta/2)/2,
              r'$\beta$',fontweight='semibold')

    axes.annotate('', xy=(r3*np.cos(theta), r3*np.sin(theta)),
                  xytext=(0,0),
                  arrowprops=dict(arrowstyle='->'))
    axes.plot(r3*np.cos(a3)/2, r3*np.sin(a3)/2, color='black',
              ls='--', lw=1)
    axes.text(r3*np.cos(theta/2)/2, 1.1*r3*np.sin(theta/2)/2,
              r'$\theta$',fontweight='semibold')


    xlim = np.max([z1[0], z2[0], np.abs(r3*np.cos(theta))])
    ylim = np.max([[z1[1], z2[1], np.abs(r3*np.sin(theta))]])
    if xlim > ylim:
        ylim = xlim
    else:
        xlim = ylim

    axes.set_xlim(-1.1*xlim, 1.1*xlim)
    axes.set_ylim(-1.1*ylim, 1.1*ylim)
    # plt.savefig(f'{file_name}.jpg', dpi=300)


def plot_complex_root(z, n):
    """
    Input (x,y): a pair of number, z = x + j * y
    Input n: find the nth root
    Return : All the complex number that statisy the equation: (root)^n = z
    """
    print(f'Trying to find the n root of complexnumber z= {z[0]} + j{z[1]}')
    x = z[0]
    y = z[1]
    r = np.sqrt(x**2 + y**2)
    if np.abs(x) < 1e-9:
        theta = np.pi / 2
    elif x < 0:
        theta = np.arctan(y/x) + np.pi
    else:
        theta = np.arctan(y/x)

    roots = []
    roots_angle = []
    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw={'projection':'polar'})
    R = r**(1/n)
    for k in range(0, n):
        angle = theta + 2 * np.pi * k / n
        roots.append((R * np.cos(angle), R * np.sin(angle)))
        roots_angle.append(angle)
        ax.plot([angle, angle],[0, R], color='lightblue', linestyle='--')
        ax.scatter(angle, R, color='orange', marker='o')
        ax.text(angle-np.pi/50, 0.98 * R, f'No.{k+1} root',
                fontsize=8, fontweight=100)
    if n > 2:
        a0 = roots_angle[0]
        a1 = roots_angle[1]
        ax.annotate('',(a1, R/2), (a0, R/2),
                    arrowprops=dict(arrowstyle='->',
                                    lw=2,
                                    connectionstyle='arc3, rad=0.35'))
        ax.text((a0+a1)/2,0.55*R, r'$\pi /$'+f'{n}')
    ax.set(frame_on=False)
    ax.set_rticks([0,R/2,R])
    ax.set_thetagrids([0,90,180,270],
                      labels=[r'$0$',r'$\pi /2$',r'$\pi$',r'$3\pi /2$'])
    ax.set_title(f'Roots of complex number: $z={z[0]} + j{z[1]}$',
                 fontweight='semibold')
    ax.grid(True, color='lightgrey', lw=1, alpha=0.5)

    # plt.savefig(f'{file_name}.jpg', dpi=300)
    plt.show()

    return np.array(roots)

if __name__ == "__main__":
    z1 = (3, np.sqrt(27))
    z2 = (-np.sqrt(3), 1)
    print(f'Plot complex number:({z1[0]},{z1[1]})')
    plot_complex(z1)
    plt.show()
    print(f'Plot complex number:({z2[0]},{z2[1]})')
    plot_complex(z2)
    plt.show()
    print(f"""Plot addtion of complex number ({z1[0]}, {z1[1]})
            and ({z2[0]}, {z2[1]})""")
    plot_complex_addition(z1, z2)
    plt.show()
    print(f"""Plot substraction of complex number ({z2[0]}, {z2[1]})
            from ({z1[0]}, {z1[1]})""")
    plot_complex_substraction(z1, z2)
    plt.show()
    print(f"""Plot multiplication of complex number ({z1[0]}, {z1[1]})
            from ({z2[0]}, {z2[1]})""")
    plot_complex_multiply(z1, z2)
    plt.show()
    print(f"""Plot division of complex number ({z1[0]}, {z1[1]})
            by ({z2[0]}, {z2[1]})""")
    plot_complex_division(z1, z2)
    plt.show()
    plot_complex_root((3, 4), 10)
