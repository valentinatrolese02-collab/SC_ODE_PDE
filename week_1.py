# Write a small Python program to compute an approximation to the second spatial deriva
# tive of a function u(x) = exp(cos(x)) at the grid point x = 0 on an equidistant grid.
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

def fdcoeffV(k, xbar, x):

    n = len(x)
    x = np.array(x)
    xrow = x - xbar

    # Initialize matrix A with ones
    A = np.ones((n, n))

    for i in range(1, n):
        A[i, :] = (xrow ** i) / factorial(i)
        
    # b is the right-hand side vector
    b = np.zeros(n)
    b[k] = 1
    
    # Solve the linear system A * c = b 
    c = np.linalg.solve(A, b)
    
    return c

def u(x):
    return np.exp( np.cos(x) )

xbar = 0 # grid point
h = 0.05
x = [xbar + alpha*h for alpha in range(-2, 2)] # assume a centered grid 
K = 2

coeff = fdcoeffV(k = 2, xbar = xbar,  x = x)
approx_d2u_dx2 = np.sum(coeff * u(x))

print(approx_d2u_dx2)

""" Then, carry out a convergence test where errors are measured as a function of the grid
increment h. Demonstrate in a plot that the correct rate of convergence is obtained for
h →0, using, e.g., the grid increments h = 1/2**s, for s = 2,3,4,5,.... What can be said
about the behavior of the truncation error for h large vs. h → 0? (HINT: you may find
that there is a difference between the theoretical convergence rate and the one achieved
when running the code... explain that then!)"""

def _2nd_der_u(x): 
    return u(x)*((np.sin(x))**2 - np.cos(x))

hs = [0.5**s for s in range(2, 8)]
# hs = [0.5, 0.7, 0.9, 1.1, 1.5]
errors = []

for h in hs:
    #x = [ xbar + alpha*h for alpha in range(-2, 3)]
    x = [xbar - 0.5*h + alpha*h for alpha in range(4)] # forward shifted grid with 3 points

    coeff = fdcoeffV(k = 2, xbar = xbar,  x = x)
    approx_d2u_dx2= np.sum(coeff * u(x)) # approximated derivative around xbar

    trunc_error = abs(_2nd_der_u(xbar) - approx_d2u_dx2)
    errors.append(trunc_error)

# Plot
# plt.figure(figsize=(7,5))
# plt.loglog(hs, errors, 'o-', label="Error")
# plt.loglog(hs, [h**2 for h in hs], '--', label="Reference slope h^3")
# plt.xlabel("h")
# plt.ylabel("Error")
# plt.legend()
# plt.title("Convergence of 3-point stencil for u''(0)")
# plt.show()

"""
Determine a stencil for interpolation of the function from c) at x = 0 that has an order of
accuracy of 3. Demonstrate the expected correct convergence rate in a convergence test.
Use a uniform grid where grid distances are of size h between nodes and that contains
two nodes symmetric around x = 0 with such nodes next to this expansion point at a
distance h/2 away from x = 0 at each side, for example a 2-point grid for interpolation
would then be consisting of x0−h/2 and x0+h/2. How many grid points do we need for
the interpolation at order 3?
"""

def u(x):
    return np.exp( np.cos(x) )

def u_prime_exact(x):
    return -np.sin(x) * np.exp(np.cos(x))

xbar = 0 # grid point
h = 0.05
x = [ xbar + alpha * h for alpha in range ( -1 , 2) ] # assume a centered grid

coeff = fdcoeffV ( k = 1 , xbar = xbar , x = x )
approx_du_dx = np . sum ( coeff * u ( x ) ) # approximated 1st derivative

def grad_richardson(f, xbar, h, p, q):
    """ Compute the Richardson extrapolation estimate of the gradient of f at x using step size h.  
    f: function to differentiate
    x: point at which to compute the gradient
    h: step size
    p: order of the method used to compute the initial estimates
    q: order of the derivative approximation (number of points in the stencil)
    """
    num_points = q + 2 # Ensure we have enough points
    limit = num_points // 2
    indices = np.arange(-limit, limit + 1)
    xh = xbar + indices * h
    
    xh2 = xbar + indices * (h / 2)

    coeffs_h = fdcoeffV(k=q, xbar=xbar, x=xh)
    coeffs_h2   = fdcoeffV(k=q, xbar=xbar, x=xh2)
    
    D_h   = np.sum(coeffs_h * f(xh))
    D_h_2 = np.sum(coeffs_h2   * f(xh2))

    # Compute the Richardson extrapolation estimate
    gradientestimate = (2**p * D_h_2 - D_h) / (2**p - 1)
    
    return gradientestimate, D_h



x_target = 1.0
exact = u_prime_exact(x_target)
hs = [0.5**s for s in range(2, 8)]

errors_base = []
errors_re = []

for h in hs:
    val_re, val_base = grad_richardson(u, x_target, h, p=2, q=1)
    
    errors_base.append(abs(val_base - exact))
    errors_re.append(abs(val_re - exact))

    print(f"h={h:.5f}, Base Error={errors_base[-1]:.2e}, RE Error={errors_re[-1]:.2e}")

plt.figure(figsize=(7, 5))

plt.loglog(hs, errors_base, 'o-', label='Base Method (Central Diff)', linewidth=2)
plt.loglog(hs, errors_re, 's-', label='Richardson Extrapolation', linewidth=2)

plt.loglog(hs, [h**2 for h in hs], '--', label="Reference slope h^2")

plt.loglog(hs, [h**4 for h in hs], '--', label="Reference slope h^4")

plt.xlabel('Step size $h$ (log scale)')
plt.ylabel('Absolute Error (log scale)')
plt.title('Convergence Analysis: Central Difference vs Richardson Extrapolation')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
plt.show()