import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Plot the function tanh(x) and -2*k/(1-3*x^2)

def fun(x):
    return -2*x / (1 - 3 * x**2)


def objFun(x):
    return np.tanh(x) - fun(x)
    
    
funV = np.vectorize(fun)


# Domain:[0, 1/sqrt(3))
td = np.linspace(0, 4)
# Domain:(1/sqrt(3), 4]
delta = 0.05
d1 = np.linspace(0 , 1/np.sqrt(3) - delta, 500)
d2 = np.linspace(1/np.sqrt(3) + delta, 4, 500)

y1 = np.tanh(td)
y21 = funV(d1)
y22 = funV(d2)

x0 = optimize.newton(objFun, 1.2)
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(td, y1, color='orange')

ax.plot(d1, y21, color='lightblue')
ax.plot(d2, y22, color='lightblue')
ax.plot([1/np.sqrt(3)]*10, np.linspace(-4,4,num=10),
        ls='--', color='black')
ax.scatter(x0, np.tanh(x0), marker='X',
           color='red',s=36)
ax.annotate(r'$y=\tanh(x)$', (td[-1]-0.5,np.tan(td[-1])),
            (td[-1]-0.5,np.tan(td[-1])+1),
            arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3,rad=0')
            )
ax.annotate(r'$y=\frac{-2x}{1-3x^2}$', (0.8,fun(0.8)),
            (0.8+1,fun(0.8)),
            arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3,rad=0')
            )
ax.annotate(f'({x0:.1f}, {np.tanh(x0):.1f})',
            (x0, np.tanh(x0)),
            (x0, np.tanh(x0)-3),
            arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc, rad=0',
                            ls='--')
            )            

            
ax.set_xlabel(r'$x$', loc='right', fontsize=12)
ax.set_xlim(0, 4)
ax.set_ylabel(r'$y$', loc='top', fontsize=12,
              rotation=0, labelpad=-30)
ax.set_ylim(-4, 4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data',0))
plt.savefig('sl1.jpg', dpi=300)
plt.show()

# Plot the function tan(x) and -2*k/(1+3*k^2)
delt = np.pi/10
num = 500
d = [np.linspace(-np.pi/2 + k*np.pi+delt, 
                 np.pi/2 + k*np.pi-delt,num=num) 
     for k in range(1,4)]
d.insert(0, np.linspace(0, np.pi/2, num=num))
d.append(np.linspace(3.5*np.pi+delt, 
                     4*np.pi, num=num))

def gFun(x):
    return -2 * x / (1 + 3 * x**2)


def objFun(x):
    return np.tan(x) - gFun(x)

gFunV = np.vectorize(gFun)

fig, ax = plt.subplots(figsize=(12,6))

for i in range(len(d)):
    x = d[i]
    y1 = np.tan(x)
    x0 = optimize.newton(objFun, i*np.pi)
    ax.plot(x, y1, color='orange')
    ax.scatter(x0, np.tan(x0), marker='X',
               color='red', s=50)
    if i > 0:
        ax.plot([i*np.pi]*10, np.linspace(-4,4,10),
                ls='--',color='black') 
x2 = np.linspace(0, 4*np.pi, num=500)
ax.plot(x2, gFunV(x2), color='lightblue')
ax.annotate(r'$y=\tan(x)$',
            (1.2, np.tan(1.2)),
            (1.2+0.5, np.tan(1.2)),
            arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3, rad=0'))
ax.annotate(r'$y=\frac{-2x}{1+3x^2}$',
            (0.5, gFun(0.5)),
            (0.5, gFun(0.5)-1),
            arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3, rad=0'))

ax.set_xlabel(r'$x$', loc='right', fontsize=12)
ax.set_xlim(-1, 4*np.pi+1)
ax.set_ylabel(r'$y$', loc='top', fontsize=12,
              rotation=0, labelpad=-30)
ax.set_ylim(-3, 3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data',0))
plt.savefig('sl2.jpg', dpi=300)
plt.show()

