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

hs = [0.5**s for s in range(2, 14)] #test for small h
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
plt.loglog(hs, errors[0] * (hs/np.array(hs)[0])**4, '--', label='$h^4$ reference')
plt.xlabel("h")
plt.ylabel("Error")
plt.grid(True, which='both', ls='--', alpha=0.5)
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

def compute_errors(num_points, test_xbar):
    errors = []
    start_alpha = -(num_points // 2) + 0.5 
    
    for h in hs:
        x = [test_xbar + (start_alpha + i)*h for i in range(num_points)]
        coeff = fdcoeffV(k=0, xbar=test_xbar, x=x) 
        approx_u = np.sum(coeff * u(x)) 
        errors.append(abs(u(test_xbar) - approx_u)) 
        
    return np.asarray(errors)

# Calculate errors for x = 0 (Anomaly)
TE2_0 = compute_errors(2, 0.0)
TE3_0 = compute_errors(3, 0.0)
TE4_0 = compute_errors(4, 0.0)

# Calculate errors for x = 0.3 (Theoretical behavior)
TE2_neq = compute_errors(2, 0.3)
TE3_neq = compute_errors(3, 0.3)
TE4_neq = compute_errors(4, 0.3)

# Plotting side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: x = 0
ax1.loglog(hs, TE2_0, 'o-', label="TE with 2 points")
ax1.loglog(hs, TE3_0, 's-', label="TE with 3 points")
ax1.loglog(hs, TE4_0, '^-', label="TE with 4 points")
ax1.loglog(hs, TE2_0[0] * (hs/np.array(hs)[0])**2, '--', color='gray', label="Ref $h^2$")
ax1.loglog(hs, TE4_0[0] * (hs/np.array(hs)[0])**4, '--', color='black', label="Ref $h^4$")
ax1.set_title("Evaluation at x = 0 (Superconvergence)")
ax1.set_xlabel("h")
ax1.set_ylabel("Truncation Error (TE)")
ax1.grid(True, which='both', ls='--', alpha=0.5)
ax1.legend()

# Subplot 2: x = 0.3
ax2.loglog(hs, TE2_neq, 'o-', label="TE with 2 points")
ax2.loglog(hs, TE3_neq, 's-', label="TE with 3 points")
ax2.loglog(hs, TE4_neq, '^-', label="TE with 4 points")
ax2.loglog(hs, TE2_neq[0] * (hs/np.array(hs)[0])**2, '--', color='gray', label="Ref $h^2$")
ax2.loglog(hs, TE3_neq[0] * (hs/np.array(hs)[0])**3, '--', color='red', label="Ref $h^3$")
ax2.loglog(hs, TE4_neq[0] * (hs/np.array(hs)[0])**4, '--', color='black', label="Ref $h^4$")
ax2.set_title("Evaluation at x = 0.3 (Theoretical behavior)")
ax2.set_xlabel("h")
ax2.grid(True, which='both', ls='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig("figure1e.svg", format="svg")
plt.show()
"""
(g) Write a small Python program to compute approximations to the first spatial derivative
of a function u(x) = exp(cos(x)) at the grid point x = 0 on an equidistant grid using
either a central derivative or Richardson extrapolation, and demonstrate in a convergence
test that expected convergence rates are achieved
"""

def _1st_der_u(x):
    return -np.sin(x) * np.exp(np.cos(x))

# Central derivative operator
def central_der(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Richardson extrapolation for the first spatial derivative
def richardson(f, x, h):
    D_h  = central_der(f, x, h)
    D_h2 = central_der(f, x, h / 2)
    return (4 * D_h2 - D_h) / 3   # Acceleration for p = 2

# Convergence test setup
hs = np.array([1/2**s for s in range(2, 12)])
points = [0.0, 1.0]

# Initialize subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, x_val in enumerate(points):
    exact = _1st_der_u(x_val)
    
    errors_CD = []
    errors_RE = []
    
    for h in hs:
        CD = central_der(u, x_val, h)
        RE = richardson(u, x_val, h)
        
        err_CD = abs(CD - exact)
        err_RE = abs(RE - exact)
        
        # Clamp errors to machine epsilon to avoid log(0) issues on the plot
        errors_CD.append(np.maximum(err_CD, 1e-16))
        errors_RE.append(np.maximum(err_RE, 1e-16))
        
    # Plotting on the respective subplot
    ax = axes[idx]
    ax.loglog(hs, errors_CD, 'o-', label='Central Difference (order 2)')
    ax.loglog(hs, errors_RE, 's-', label='Richardson Extrapolation (order 4)')
    
    # Reference slopes (only plot if the error is well above machine precision)
    if errors_CD[0] > 1e-15:
        ax.loglog(hs, errors_CD[0] * (hs / hs[0])**2, '--', label='$h^2$ reference')
    if errors_RE[0] > 1e-15:
        ax.loglog(hs, errors_RE[0] * (hs / hs[0])**4, '--', label='$h^4$ reference')

    ax.invert_xaxis()
    ax.set_xlabel('Step size $h$ (log scale)')
    ax.set_ylabel('Absolute Error (log scale)')
    ax.set_title(f'Evaluation at x = {x_val}')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

plt.suptitle('Convergence Analysis: Central Difference vs Richardson Extrapolation', fontsize=14)
plt.tight_layout()
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
