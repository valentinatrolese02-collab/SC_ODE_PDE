"""Write a code implementing the five-point Laplacian scheme. 
This applies to POisson pb: uxx + uyy = f, on [0,1]x[0,1] with 
Dirichlet boundary conditions. Check that you get the
expected order of convergence, e.g. on a problem with the exact solution
uexact(x, y) = sin(4π(x + y)) + cos(4πxy)"""

import numpy as np
from scipy.sparse import spdiags, eye, kron
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from typing import Literal


# function suggested from slides
def poisson5(m):
    e = np.ones(m)
    S = spdiags([e, -2*e, e], [-1, 0, 1], m, m)
    I = eye(m, format='csr')
    A = kron(I, S) + kron(S, I)
    # h = 1 / (m + 1)
    A = (m + 1)**2 * A #scaling from grid spacing
    return A

def poisson9(m):
    e = np.ones(m)
    S = spdiags([-e, -10*e, -e], [-1, 0, 1], m, m)
    I = spdiags([-0.5*e, e, -0.5*e], [-1, 0, 1], m, m)
    A = (1/6) * (m + 1)**2 * (kron(I, S) + kron(S, I))
    return A

def Laplacian_scheme(
        f, # function to solve
        g, # function for Dirichlet boundary conditions
        m: int = 10, # number of interior points
        n: Literal[5,9] = 5 # default -5 points Poisson
):

    # full grid including boundary
    x = np.linspace(0, 1, m+2)
    y = np.linspace(0, 1, m+2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    h2_inv = (m + 1)**2 # 1/h^2
    
    # f on interior points (vectorized)
    F = f(X[1:-1, 1:-1], Y[1:-1, 1:-1]).reshape(m*m)

    # Correct RHS = Right Hand-Side: function we are going to correct by
    # adding g evaluated on boundary poins, accordingly to the laplancian scheme used
    RHS = F.copy()

    # choose stencil
    if n == 5:
        A = poisson5(m)

        # bottom boundary: fixed i=0, variate y from j=1 to m
        bottom = g(x[0], y[1:-1])
        RHS[:m] -= h2_inv*bottom

        # top boundary: i=m+1, j=1..m
        top = g(x[-1], y[1:-1])
        RHS[-m:] -= h2_inv*top

        # left boundary: i=1..m, j=0
        left = g(x[1:-1], y[0])
        RHS[::m] -= h2_inv*left

        # right boundary: i=1..m, j=m+1
        right = g(x[1:-1], y[-1])
        RHS[m-1::m] -= h2_inv*right
    
    else:
        A = poisson9(m)
        # bottom boundary: fixed i=0, variate y from j=1 to m
        bottom = g(x[0], y[1:-1])
        RHS[:m] += 4*bottom 

        # top boundary: i=m+1, j=1..m
        top = g(x[-1], y[1:-1])
        RHS[-m:] += top

        # left boundary: i=1..m, j=0
        left = g(x[1:-1], y[0])
        RHS[::m] += left

        # right boundary: i=1..m, j=m+1
        right = g(x[1:-1], y[-1])
        RHS[m-1::m] += right

    return x, y, A, RHS.flatten() # outpus also the grid


# Test function
u = lambda x, y: np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)
u_xx = lambda x, y: - 16* np.pi**2 * (np.sin(4 * np.pi * (x + y)) + y**2 * np.cos(4 * np.pi * x * y))
u_yy = lambda x, y: - 16* np.pi**2 * (np.sin(4 * np.pi * (x + y)) + x**2 * np.cos(4 * np.pi * x * y))
f = lambda x, y: u_xx(x,y) + u_yy(x,y)

# N values chosen so that 4000 is perfectly divisible by N+1
m_list = [49, 99, 199, 399]
h_values = []
errors = []

print("\nComputing test solutions...")
for m in m_list:
    x, y, A, b = Laplacian_scheme(f, f, m = m)
    u_vec = spsolve(A, b) # A sparse
    
    U = np.zeros((m+2, m+2))
    U[1:-1, 1:-1] = u_vec.reshape(m, m)
    # Complete grid
    X, Y = np.meshgrid(x, y, indexing='ij')
    U_exact = u(X, Y)

    error = np.max(np.abs(U - U_exact)) # L infty norm
    h = 1/(m+1)
    h_values.append(h)
    errors.append(error)

# --- 3. Estimate Convergence Order (p) ---
# Fit a line to the log-log data: log(E) = p * log(h) + log(C)
log_h = np.log(h_values)
log_E = np.log(errors)
p_slope, log_C = np.polyfit(log_h, log_E, 1)

print(f"\nEstimated global order of convergence (slope): p = {p_slope:.4f}")

# --- 4. Plot the Convergence Figure ---
plt.figure(figsize=(8, 6))

# Plot actual errors
plt.loglog(
    h_values,
    errors,
    marker='o',
    linestyle='-',
    linewidth=2,
    label=f'Numerical Error (slope ≈ {p_slope:.2f})'
)

# Reference O(h^2) line
h_ref_line = np.array([h_values[0], h_values[-1]])
C_ref = errors[0] / (h_values[0]**2)
error_ref_line = C_ref * (h_ref_line**2)

plt.loglog(
    h_ref_line,
    error_ref_line,
    linestyle='--',
    color='gray',
    label='Reference $O(h^2)$'
)

# Formatting
plt.xlabel('Grid spacing $h$ (log scale)')
plt.ylabel('Global Error $E(h)$ (log scale)')
plt.title('Convergence of the Global Error')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()


