import sympy
import numpy as np
from scipy import special
from scipy import optimize
import matplotlib.pyplot as plt


line_len = 50
x, a, lam = sympy.symbols('x, a, lamda')
y = sympy.Function('y')
u = sympy.Function('u')
ode = sympy.Eq(-y(x).diff(x, 2) - (1 + x) * y(x), lam * y(x))
sol = sympy.dsolve(ode, y(x))
print(f'General solutions for the differential equations:')
sympy.pprint(sol)
print('-'*line_len)

boundary_eq1 = sol.rhs.diff(x).subs(x, 0)
boundary_eq2 = sol.rhs.subs(x, 2)

print(f'Imposing boundary condition at x=0:')
sympy.pprint(boundary_eq1)
print('-'*line_len)
print(f'Imposing boundary condition at x=2:')
sympy.pprint(boundary_eq2)
print('-'*line_len)



def plot_determinant():
    Lam = np.linspace(0, 16, 100)
    ai, _, bi, _ = special.airy(-3 - Lam)
    _, ai_prime, _, bi_prime = special.airy(-1 - Lam)
    determinant = ai_prime * bi - ai * bi_prime
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(Lam, determinant, color='lightblue')
    ax.set_xlabel(r'$\lambda$', loc='right')
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_ylabel(r'$Determinant$')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(r'$\lambda$' + f' and the cooresponding determinnant')
    plt.show()

def root_minimum():
    def f(x):
        ai, _, bi, _ = special.airy(-3 - x)
        _, ai_prime, _, bi_prime = special.airy(-1 - x)
        return ai_prime * bi - bi_prime * ai
    zero = optimize.brentq(f, 2, 4)
    return zero

def find_rayleigh_quotient(expr_lst, bounds):

    numerator = 0
    denominator = 0

    for i in range(len(expr_lst)):
        expr = expr_lst[i]
        bound = bounds[i]
        n = sympy.integrate((expr.diff(x))**2 - (1+x)*(expr)**2,
                            (x,bound[0],bound[1]))
        numerator += n
        d = sympy.integrate(expr**2, (x, bound[0], bound[1]))
        denominator += d


    return numerator / denominator


 # Call function to plot the lambda and corresponding determinant.
plot_determinant()
print(f'The principal eigenvalue: {root_minimum()}')
print('-'*line_len)

# Define the trial functions. Keep in mind to define the boudary as well.
trial_functions = [[x**2 * (x - 2)],
                   [x**2, 2 - x],
                   [x * sympy.sin(sympy.pi * x / 2)]]
bounds = [[(0,2)],
          [(0, 1), (1, 2)],
          [(0, 2)]]

for i in range(len(trial_functions)):
    fun = trial_functions[i]
    bound = bounds[i]
    rayleigh_quotient = find_rayleigh_quotient(fun, bound)
    sympy.pprint(f'''Rayleigh quotient for the number {i+1} trial function:
          {rayleigh_quotient},
          with numerical value:{rayleigh_quotient.evalf():.5f}''')
    print('-'*line_len)
