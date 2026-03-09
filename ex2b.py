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
    
    h2_inv = (m + 1)**2 # 1/h^2

    # full grid including boundary
    x = np.linspace(0, 1, m+2)
    y = np.linspace(0, 1, m+2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    
    # f on interior points (vectorized)
    F = f(X[1:-1, 1:-1], Y[1:-1, 1:-1])

    # Correct RHS = Right Hand-Side: function we are going to correct by
    # adding g evaluated on boundary poins, accordingly to the laplancian scheme used
    RHS = F.copy()

        # Corrected indexing for 2D RHS array
    if n == 5:
        A = poisson5(m)

        # x = 0 boundary (First row of interior points)
        bottom = g(x[0], y[1:-1])
        RHS[0, :] -= h2_inv * bottom

        # x = 1 boundary (Last row of interior points)
        top = g(x[-1], y[1:-1])
        RHS[-1, :] -= h2_inv * top

        # y = 0 boundary (First column of interior points)
        left = g(x[1:-1], y[0])
        RHS[:, 0] -= h2_inv * left

        # y = 1 boundary (Last column of interior points)
        right = g(x[1:-1], y[-1])
        RHS[:, -1] -= h2_inv * right

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


# --- TESTING & VALIDATION ZONE ---
m = 20  
a, b = 0, 1

# Boundary condition g(x,y) based on exact solution
def g(x, y):
    return np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)

# Right-hand side f(x,y) obtained from the analytical Laplacian (u_xx + u_yy)
def f(x, y):
    u_xx = -16 * np.pi**2 * np.sin(4 * np.pi * (x + y)) - 16 * np.pi**2 * y**2 * np.cos(4 * np.pi * x * y)
    u_yy = -16 * np.pi**2 * np.sin(4 * np.pi * (x + y)) - 16 * np.pi**2 * x**2 * np.cos(4 * np.pi * x * y)
    return u_xx + u_yy

# Solve system for both stencils
x, y, A, u_5_flat = Laplacian_scheme(f, g, m, 5)

# Rigorous numerical verification against exact interior solution
X, Y = np.meshgrid(x, y)

u_exact_flat = g(X, Y).flatten()

error_5 = np.max(np.abs(u_exact_flat - u_5_flat))

print(f"Results for m={m}:")
print(f"Max Error (5-point): {error_5:.2e}")

# ####################### try convergence error ##################################

# List of grid sizes to test
m_values = [10, 20, 40, 80]

errors_5 = []
errors_9 = []
h_values = []

print(f"{'m':>4} | {'h':>8} | {'Err 5-pts':>11} | {'Ord 5':>5} | {'Err 9-pts':>11} | {'Ord 9':>5}")
print("-" * 65)

for i, m in enumerate(m_values):
    h = (b - a) / (m + 1)
    h_values.append(h)
    
    # Solve for both schemes
    x, y, A, u_5 = Laplacian_scheme(f, g, m, 5)

    
    # Calculate the exact solution on the interior points
    x_int = np.linspace(a, b, m+2)[1:-1]
    y_int = np.linspace(a, b, m+2)[1:-1]
    X_int, Y_int = np.meshgrid(x_int, y_int)
    u_exact = g(X_int, Y_int).flatten()
    
    # Maximum error (Infinity norm)
    e5 = np.max(np.abs(u_exact - u_5))
    
    errors_5.append(e5)
    
    # Calculate the empirical order of convergence relative to the previous m
    if i == 0:
        ord_5, ord_9 = 0.0, 0.0
    else:
        ord_5 = np.log(errors_5[i-1] / errors_5[i]) / np.log(h_values[i-1] / h_values[i])
        
    print(f"{m:4d} | {h:.6f} | {e5:.5e} | {ord_5:5.2f}")

# --- LOG-LOG PLOT (Ideal for the report) ---
plt.figure(figsize=(8, 6))
plt.loglog(h_values, errors_5, 'o-', label='5-point error')

# Reference lines for O(h^2) and O(h^4) slopes
ref_h = np.array(h_values)
plt.loglog(ref_h, errors_5[-1] * (ref_h / ref_h[-1])**2, 'k--', label='$\mathcal{O}(h^2)$ reference')

plt.xlabel('Grid spacing (h)')
plt.ylabel('Max Error (Infinity Norm)')
plt.title('Convergence of 5-point and 9-point Laplacian Schemes')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()