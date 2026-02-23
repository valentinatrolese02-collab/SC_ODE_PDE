import numpy as np
from scipy.sparse import spdiags, eye, kron
import matplotlib.pyplot as plt
from typing import Literal

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

def Laplacian(m, f, g, a, b, n: Literal[5,9]):
    h = (b - a) / (m + 1)
    x = np.linspace(a, b, m+2)
    y = np.linspace(a, b, m+2)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate full matrices and extract interior points for f
    F_full = f(X, Y)
    F = F_full[1:-1, 1:-1]
    
    # Create a full-size grid filled with boundary conditions (g).
    U_bnd = np.zeros_like(F_full)
    U_bnd[0, :] = g(X[0, :], Y[0, :])       # Bottom boundary
    U_bnd[-1, :] = g(X[-1,:],Y[-1,:])     # Top boundary
    U_bnd[:, 0] = g(X[:, 0], Y[:, 0])       # Left boundary
    U_bnd[:, -1] = g(X[:, -1], Y[:, -1])     # Right boundary

    if n == 5:
        # Extract boundary contributions by shifting the grid (Up, Down, Left, Right)
        bnd_contrib = (
            U_bnd[0:-2, 1:-1] + U_bnd[2:, 1:-1] + 
            U_bnd[1:-1, 0:-2] + U_bnd[1:-1, 2:]
        )
        F_copy = F - bnd_contrib * (m + 1)**2
        
        A = poisson5(m)
        u = np.linalg.solve(A.toarray(), F_copy.flatten())
        
    elif n == 9:
        # Deferred correction for right-hand side to achieve O(h^4) accuracy
        lap_f_unscaled = (F_full[0:-2, 1:-1] + F_full[2:, 1:-1] + 
                          F_full[1:-1, 0:-2] + F_full[1:-1, 2:] - 
                          4 * F)
        F_corrected = F + lap_f_unscaled / 12.0
        
        # Boundary contributions mapped directly via stencil weights:
        # Straight neighbors (weight 4) + Diagonal neighbors (weight 1)
        bnd_contrib = (
            # 1. DIRECT NEIGHBORS (Weight = 4)
            4 * U_bnd[0:-2, 1:-1] +  # Neighbor BELOW (Y-1, X)
            4 * U_bnd[2:, 1:-1]   +  # Neighbor ABOVE (Y+1, X)
            4 * U_bnd[1:-1, 0:-2] +  # Neighbor LEFT  (Y, X-1)
            4 * U_bnd[1:-1, 2:]   +  # Neighbor RIGHT (Y, X+1)
            
            # 2. DIAGONAL NEIGHBORS (Weight = 1)
            1 * U_bnd[0:-2, 0:-2] +  # Corner BOTTOM-LEFT  (Y-1, X-1)
            1 * U_bnd[2:, 0:-2]   +  # Corner TOP-LEFT     (Y+1, X-1)
            1 * U_bnd[0:-2, 2:]   +  # Corner BOTTOM-RIGHT (Y-1, X+1)
            1 * U_bnd[2:, 2:]        # Corner TOP-RIGHT    (Y+1, X+1)
        )
        
        # Scale boundary contribution by 1 / (6*h^2) and subtract from RHS
        F_corrected -= bnd_contrib * ((m + 1)**2) / 6.0  
        
        A = poisson9(m)
        u = np.linalg.solve(A.toarray(), F_corrected.flatten())
        
    else:
        raise ValueError("n must be either 5 or 9.")

    return u

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
u_5_flat = Laplacian(m, f, g, a, b, 5)
u_9_flat = Laplacian(m, f, g, a, b, 9)

# Rigorous numerical verification against exact interior solution
x_int = np.linspace(a, b, m+2)[1:-1]
y_int = np.linspace(a, b, m+2)[1:-1]
X_int, Y_int = np.meshgrid(x_int, y_int)

u_exact_flat = g(X_int, Y_int).flatten()

error_5 = np.max(np.abs(u_exact_flat - u_5_flat))
error_9 = np.max(np.abs(u_exact_flat - u_9_flat))

print(f"Results for m={m}:")
print(f"Max Error (5-point): {error_5:.2e}")
print(f"Max Error (9-point): {error_9:.2e}")

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
    u_5 = Laplacian(m, f, g, a, b, 5)
    u_9 = Laplacian(m, f, g, a, b, 9)
    
    # Calculate the exact solution on the interior points
    x_int = np.linspace(a, b, m+2)[1:-1]
    y_int = np.linspace(a, b, m+2)[1:-1]
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

# --- LOG-LOG PLOT (Ideal for the report) ---
plt.figure(figsize=(8, 6))
plt.loglog(h_values, errors_5, 'o-', label='5-point error')
plt.loglog(h_values, errors_9, 's-', label='9-point error')

# Reference lines for O(h^2) and O(h^4) slopes
ref_h = np.array(h_values)
plt.loglog(ref_h, errors_5[-1] * (ref_h / ref_h[-1])**2, 'k--', label='$\mathcal{O}(h^2)$ reference')
plt.loglog(ref_h, errors_9[-1] * (ref_h / ref_h[-1])**4, 'r--', label='$\mathcal{O}(h^4)$ reference')

plt.xlabel('Grid spacing (h)')
plt.ylabel('Max Error (Infinity Norm)')
plt.title('Convergence of 5-point and 9-point Laplacian Schemes')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()


# ########################### Proves #################

# F_full = f(X, Y)

# F = F_full[1:-1, 1:-1]



# if n == 5:

# # Incorporate boundary conditions (the contour) ONLY on the interior edges

# # Bottom and Top boundaries (1, end-1)

# F[0, :] -= g(X[0, 1:-1], Y[0, 1:-1]) * (m + 1)**2

# F[-1, :] -= g(X[-1, 1:-1], Y[-1, 1:-1]) * (m + 1)**2


# # Left and Right boundaries (1,end-1)

# F[:, 0] -= g(X[1:-1, 0], Y[1:-1, 0]) * (m + 1)**2

# F[:, -1] -= g(X[1:-1, -1], Y[1:-1, -1]) * (m + 1)**2

# F = F.flatten() # Flatten the modified F5 for the solver

# A = poisson5(m)

# u = np.linalg.solve(A.toarray(), F)

# elif n == 9:


# # Compute unscaled 5-point Laplacian of f (h^2 cancels out mathematically)

# lap_f_unscaled = (F_full[0:-2, 1:-1] + F_full[2:, 1:-1] +

# F_full[1:-1, 0:-2] + F_full[1:-1, 2:] -

# 4 * F)


# # Apply correction: F_corrected = f_ij + (h^2 / 12) * \nabla_5^2 f_ij

# F_corrected = F + lap_f_unscaled / 12.0


# # 2. Apply Dirichlet Boundary Conditions (g)

# # Scaling factor for the 9-point matrix is C = 1 / (6 * h^2)

# C = ((m + 1)**2) / 6.0


# # Direct neighbors (cross) have weight 4

# F_corrected[0, :] -= 4 * g(X[0, 1:-1], Y[0, 1:-1]) * C # Bottom

# F_corrected[-1, :] -= 4 * g(X[-1, 1:-1], Y[-1, 1:-1]) * C # Top

# F_corrected[:, 0] -= 4 * g(X[1:-1, 0], Y[1:-1, 0]) * C # Left

# F_corrected[:, -1] -= 4 * g(X[1:-1, -1], Y[1:-1, -1]) * C # Right


# # Diagonal neighbors (corners) have weight 1

# F_corrected[0, 0] -= 1 * g(X[0, 0], Y[0, 0]) * C # Bottom-Left

# F_corrected[-1, 0] -= 1 * g(X[-1, 0], Y[-1, 0]) * C # Top-Left

# F_corrected[0, -1] -= 1 * g(X[0, -1], Y[0, -1]) * C # Bottom-Right

# F_corrected[-1, -1] -= 1 * g(X[-1, -1], Y[-1, -1]) * C # Top-Right


# A = poisson9(m)

# u = np.linalg.solve(A.toarray(), F_corrected.flatten())

# else:

# raise ValueError("n must be either 5 or 9.")
