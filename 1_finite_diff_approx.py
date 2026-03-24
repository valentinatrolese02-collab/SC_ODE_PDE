""" 
(c) Write a small Python program to compute an approximation to the second spatial 
derivative of a function u(x) = exp(cos(x)) at the grid point x = 0 on an equidistant grid.
"""

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

# analytical function and its second derivative
def u(x):
    return np.exp( np.cos(x) )

def _1st_der_u(x):
    return - u(x)*np.sin(x)

def _2nd_der_u(x): 
    return u(x)*((np.sin(x))**2 - np.cos(x))

# grid point and increment
xbar = 0 
h = 0.05

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

x = [xbar + alpha*h for alpha in range(-2, 2)] # we assume a centered equidistant grid 
K = 2 # derivation order

coeff = fdcoeffV(k = 2, xbar = xbar,  x = x)
approx_d2u_dx2 = np.sum(coeff * u(x))

print(f"The approximated second spatial derivative of u at x = 0 is {approx_d2u_dx2}")

""" (d) Then, carry out a convergence test where errors are measured as a function of the grid
increment h. Demonstrate in a plot that the correct rate of convergence is obtained for
h →0, using, e.g., the grid increments h = 1/2**s, for s = 2,3,4,5,.... What can be said
about the behavior of the truncation error for h large vs. h → 0? (HINT: you may find
that there is a difference between the theoretical convergence rate and the one achieved
when running the code... explain that then!)"""

hs = [0.5**s for s in range(2, 8)] #test for small h
# hs = [0.5, 0.7, 0.9, 1.1, 1.5] #test for LARGE h
errors = []

for h in hs:
    x = [ xbar + alpha*h for alpha in range(-2, 3)]
    coeff = fdcoeffV(k = 2, xbar = xbar,  x = x)
    approx_d2u_dx2= np.sum(coeff * u(x)) # approximated derivative around xbar

    trunc_error = abs(_2nd_der_u(xbar) - approx_d2u_dx2)
    errors.append(trunc_error)

# Plot
plt.figure(figsize=(7,5))
plt.loglog(hs, errors, 'o-', label="Error")
plt.loglog(hs, [h**3 for h in hs], '--', label="Reference slope h^3")
plt.xlabel("h")
plt.ylabel("Error")
plt.legend()
plt.title("Convergence of 3-point stencil for u''(0)")
plt.show()

"""
(e) Determine a stencil for interpolation of the function from c) at x = 0 that has an order of
accuracy of 3. Demonstrate the expected correct convergence rate in a convergence test.
Use a uniform grid where grid distances are of size h between nodes and that contains
two nodes symmetric around x = 0 with such nodes next to this expansion point at a
distance h/2 away from x = 0 at each side, for example a 2-point grid for interpolation
would then be consisting of x0−h/2 and x0+h/2. How many grid points do we need for
the interpolation at order 3?
"""

def compute_errors(num_points):
    errors = [] 
    for h in hs:
        x = [xbar - 0.5*h + alpha*h for alpha in range(num_points)] 
        coeff = fdcoeffV(k=0, xbar=xbar, x=x) 
        approx_u = np.sum(coeff * u(x)) 
        errors.append(abs(u(xbar) - approx_u)) 
        
    return np.asarray(errors)
TE2 = compute_errors(2)
TE3 = compute_errors(3)
TE4 = compute_errors(4)

# Plot
plt.figure(figsize=(7,5))
plt.loglog(hs, TE2, 'o-', label="TE with 2 points")
plt.loglog(hs, TE3, 'o-', label="TE with 3 points")
plt.loglog(hs, TE4, 'o-', label="TE with 4 points")

plt.loglog(hs, [h**2 for h in hs], '--', label="Reference slope h^2")
plt.loglog(hs, [h**3 for h in hs], '--', label="Reference slope h^3")
plt.loglog(hs, [h**4 for h in hs], '--', label="Reference slope h^4")

plt.xlabel("h")
plt.ylabel("TE")
plt.legend()
plt.title("Attempts to find an interpolation of u(x) of order of accuracy 3")

plt.show()

"""
(g) Write a small Python program to compute approximations to the first spatial derivative
of a function u(x) = exp(cos(x)) at the grid point x = 0 on an equidistant grid using
either a central derivative or Richardson extrapolation, and demonstrate in a convergence
test that expected convergence rates are achieved
"""
xneq0 = 0.3
# central derivative operator
def central_der(f, x, h):
    return (f(x + h) - f(x - h)) / (2*h)

# Rihcardson extrapolation for the first spatial derivative
def richardson(f, x, h):
    D_h  = central_der(f, x, h)
    D_h2 = central_der(f, x, h/2)
    return (4*D_h2 - D_h) / 3   # p = 2

# CONVERGENCE TEST
exact = _1st_der_u(xneq0)
# we observe that the derivative in 
print(f"{'h':<10} | {'Err CD':<12} | {'Err RE':<12}")

errors_CD = []
errors_RE = []
for h in hs:
    CD = central_der(u, xneq0, h)
    RE = richardson(u, xneq0, h)
    err_CD = abs(CD - exact)
    err_RE = abs(RE - exact)
    errors_CD.append(err_CD)
    errors_RE.append(err_RE)

# --- Plot ---
plt.figure(figsize=(8,6))

plt.loglog(hs, errors_CD, 'o-', label='Central Difference (order 2)')
plt.loglog(hs, errors_RE, 's-', label='Richardson Extrapolation (order 4)')
# simple version of hs**2, not scaled and not aligned with the error
# plt.loglog(hs, hs**2, '--', label='$h^2$ reference')
# plt.loglog(hs, hs**4, '--', label='$h^4$ reference')

# Reference slopes scaled to match error values for visibility
plt.loglog(hs, errors_CD[0] * (hs/np.array(hs)[0])**2, '--', label='$h^2$ reference')
plt.loglog(hs, errors_RE[0] * (hs/np.array(hs)[0])**4, '--', label='$h^4$ reference')

plt.gca().invert_xaxis()
plt.xlabel('Step size $h$ (log scale)')
plt.ylabel('Absolute Error (log scale)')
plt.title('Convergence Analysis: Central Difference vs Richardson Extrapolation')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
# plt.savefig("Figure_Richardson.svg", format="svg")
plt.show()

"""
(h)  general formula for Richardson approximation
Test that it works by selecting the two stencils derived in (a) and demonstrate
that you can improve convergence using the Richardson Extrapolation (RE).
"""

xbar = 0 # grid point
h = 0.05

# Stencils from part (a)
# stencil of order 3 
def stencil1(h):
    return np.array([xbar + alpha*h for alpha in range(-4, 1)])
p1 = 3 # p=5
#stencil of order 4
def stencil2(h):
    return np.array([xbar + alpha*h for alpha in range(-2, 3)])
p2 = 4 # p=6

def stenc1_richardson(f, xbar, h, q):
    x_h1 = stencil1(h)
    x_h2 = stencil1(h/2)

    coeff_h1 = fdcoeffV(k=q, xbar=xbar, x=x_h1)
    coeff_h2 = fdcoeffV(k=q, xbar=xbar, x=x_h2)

    D_h   = np.sum(coeff_h1 * f(x_h1))
    D_h2  = np.sum(coeff_h2 * f(x_h2))

    D_RE = (2**p1 * D_h2 - D_h) / (2**p1 - 1)
    return D_RE, D_h

def stenc2_richardson(f, xbar, h, q):
    x_h1 = stencil2(h)
    x_h2 = stencil2(h/2)

    coeff_h1 = fdcoeffV(k=q, xbar=xbar, x=x_h1)
    coeff_h2 = fdcoeffV(k=q, xbar=xbar, x=x_h2)

    D_h   = np.sum(coeff_h1 * f(x_h1))
    D_h2  = np.sum(coeff_h2 * f(x_h2))

    D_RE = (2**p2 * D_h2 - D_h) / (2**p2 - 1)
    return D_RE, D_h

q = 2   # some derivative orders 

print(f"{'h':<10} | {'Err FD1':<12} | {'Err RE1':<12} | {'Err FD2':<12} | {'Err RE2':<12}")
for h in hs:
    RE1, FD1 = stenc1_richardson(u, xbar, h, q)
    RE2, FD2 = stenc2_richardson(u, xbar, h, q)
    exact_val = _2nd_der_u(xbar)
    err_FD1 = abs(FD1 - exact_val)
    err_RE1 = abs(RE1 - exact_val)
    err_FD2 = abs(FD2 - exact_val)
    err_RE2 = abs(RE2 - exact_val)
    
    print(f"{h:<10.4e} | {err_FD1:<12.4e} | {err_RE1:<12.4e} | {err_FD2:<12.4e} | {err_RE2:<12.4e}")