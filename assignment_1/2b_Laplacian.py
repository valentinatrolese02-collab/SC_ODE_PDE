"""
We solve point (c) and (d) together, by creating a function which takes as input the number of 
points for the laplacian scheme and converce test on both schemes on a unique loop
"""

import numpy as np
from scipy.sparse import spdiags, eye, kron
import matplotlib.pyplot as plt
from typing import Literal
from scipy.sparse.linalg import spsolve

def poisson5(m):
    e = np.ones(m)
    S = spdiags([e, -2*e, e], [-1, 0, 1], m, m)
    I = eye(m, format='csr')
    A = kron(I, S) + kron(S, I)
    A = (m + 1)**2 * A # Scaling from grid spacing
    return A

def poisson9(m):
    e = np.ones(m)
    S = spdiags([-e, -10*e, -e], [-1, 0, 1], m, m)
    I = spdiags([-0.5*e, e, -0.5*e], [-1, 0, 1], m, m)
    A = (1/6) * (m + 1)**2 * (kron(I, S) + kron(S, I))
    return A

def Laplacian(
        f, # function to solve
        g, # function for Dirichlet boundary conditions
        m: int = 10, # number of interior points
        n: Literal[5,9] = 5 # default -5 points Poisson
):
    
    x = np.linspace(0, 1, m+2)
    y = np.linspace(0, 1, m+2)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate full matrices and extract interior points for f
    F_full = f(X, Y)
    F = F_full[1:-1, 1:-1]
    
    # Create a full-size grid filled with boundary conditions from g
    U_bnd = np.zeros_like(F_full)
    U_bnd[0, :] = g(X[0, :], Y[0, :])       # Bottom = fixed index i = 0
    U_bnd[-1, :] = g(X[-1,:],Y[-1,:])     # Top
    U_bnd[:, 0] = g(X[:, 0], Y[:, 0])       # Left
    U_bnd[:, -1] = g(X[:, -1], Y[:, -1])     # Right

    if n == 5:
        # Extract boundary contributions by shifting the grid (top, bottom, Left, Right)
        # bnd_contrib has shape mxm
        bnd_contrib = U_bnd[0:-2, 1:-1] + U_bnd[2:, 1:-1] + U_bnd[1:-1, 0:-2] + U_bnd[1:-1, 2:]
        
        # correct right hand side
        F_corrected = (F - bnd_contrib * (m + 1)**2).flatten() # Flatten for solver
        
        A = poisson5(m)
        u = spsolve(A, F_corrected)
        
    elif n == 9:
        # Deferred correction for right-hand side 
        lap_f_unscaled = (F_full[0:-2, 1:-1] + F_full[2:, 1:-1] + 
                          F_full[1:-1, 0:-2] + F_full[1:-1, 2:] - 
                          4 * F)
        F_corrected = F + lap_f_unscaled / 12.0
        
        # Boundary contributions mapped directly via stencil weights:
        bnd_contrib = (
            # DIRECT NEIGHBORS (Weight = 4)
            4 * U_bnd[0:-2, 1:-1] +  # BELOW
            4 * U_bnd[2:, 1:-1]   +  # ABOVE 
            4 * U_bnd[1:-1, 0:-2] +  # LEFT  
            4 * U_bnd[1:-1, 2:]   +  # RIGHT 

            # DIAGONAL NEIGHBORS (Weight = 1)
            1 * U_bnd[0:-2, 0:-2] +  #  BOTTOM-LEFT
            1 * U_bnd[2:, 0:-2]   +  #  TOP-LEFT    
            1 * U_bnd[0:-2, 2:]   +  #  BOTTOM-RIGHT
            1 * U_bnd[2:, 2:]        # TOP-RIGHT   
        )

        # correct right hand side 
        F_corrected -= bnd_contrib * (m + 1)**2 / 6.0  
        F_corrected = F_corrected.flatten()
        
        A = poisson9(m)
        u = spsolve(A, F_corrected)
        
    else:
        raise ValueError("n must be either 5 or 9.")

    return u

# TESTING
m = 20  

# Boundary condition g(x,y) based on exact solution
def g(x, y):
    return np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)

# Right-hand side f(x,y) obtained from the analytical Laplacian (u_xx + u_yy)
def f(x, y):
    u_xx = -16 * np.pi**2 * np.sin(4 * np.pi * (x + y)) - 16 * np.pi**2 * y**2 * np.cos(4 * np.pi * x * y)
    u_yy = -16 * np.pi**2 * np.sin(4 * np.pi * (x + y)) - 16 * np.pi**2 * x**2 * np.cos(4 * np.pi * x * y)
    return u_xx + u_yy

# Solve system for both stencils
u_5_flat = Laplacian(f, g, m, 5)
u_9_flat = Laplacian(f, g, m, 9)

# numerical verification against exact interior solution
x_int = np.linspace(0, 1, m+2)[1:-1]
y_int = np.linspace(0, 1, m+2)[1:-1]
X_int, Y_int = np.meshgrid(x_int, y_int)

u_exact_flat = g(X_int, Y_int).flatten()

error_5 = np.max(np.abs(u_exact_flat - u_5_flat))
error_9 = np.max(np.abs(u_exact_flat - u_9_flat))

print(f"Results for m={m}:")
print(f"Max Error (5-point): {error_5:.2e}")
print(f"Max Error (9-point): {error_9:.2e}")

# COVERGENCE ERROR

# List of grid sizes to test
m_values = [50, 100, 200, 400]

errors_5 = []
errors_9 = []
h_values = []

print(f"{'m':>4} | {'h':>8} | {'Err 5-pts':>11} | {'Ord 5':>5} | {'Err 9-pts':>11} | {'Ord 9':>5}")
print("-" * 65)

for i, m in enumerate(m_values):
    h = 1 / (m + 1)
    h_values.append(h)
    
    # Solve for both schemes
    u_5 = Laplacian(f, g, m, 5)
    u_9 = Laplacian(f, g, m, 9)
    
    # Calculate the exact solution on the interior points
    x_int = np.linspace(0, 1, m+2)[1:-1]
    y_int = np.linspace(0, 1, m+2)[1:-1]
    X_int, Y_int = np.meshgrid(x_int, y_int)
    u_exact = g(X_int, Y_int).flatten()
    
    # Maximum error (Infinity norm)
    e5 = np.max(np.abs(u_exact - u_5))
    e9 = np.max(np.abs(u_exact - u_9))
    
    errors_5.append(e5)
    errors_9.append(e9)
    
    # Calculate the empirical order of convergence relative to the previous m
    if i == 0:
        ord_5, ord_9 = 0.0, 0.0
    else:
        ord_5 = np.log(errors_5[i-1] / errors_5[i]) / np.log(h_values[i-1] / h_values[i])
        ord_9 = np.log(errors_9[i-1] / errors_9[i]) / np.log(h_values[i-1] / h_values[i])
        
    print(f"{m:4d} | {h:.6f} | {e5:.5e} | {ord_5:5.2f} | {e9:.5e} | {ord_9:5.2f}")

#  LOG-LOG PLOT
plt.figure(figsize=(8, 6))
plt.loglog(h_values, errors_5, 'o-', label='5-point error')
plt.loglog(h_values, errors_9, 's-', label='9-point error')

# Reference lines shifted to overlay our data for better visualization
ref_h = np.array(h_values)
# h^2
plt.loglog(ref_h, errors_5[-1] * (ref_h / ref_h[-1])**2, 'k--', label=r'$\mathcal{O}(h^2)$ reference')
# h^4
plt.loglog(ref_h, errors_9[-1] * (ref_h / ref_h[-1])**4, 'r--', label=r'$\mathcal{O}(h^4)$ reference')

plt.xlabel('Grid spacing (h)')
plt.ylabel('Max Error (Infinity Norm)')
plt.title('Convergence of 5-point and 9-point Laplacian Schemes')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
