import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Parameters and analytical functions
a = 0.5
psi = lambda x: 20 * np.pi * x**3
psidot = lambda x: 3 * 20 * np.pi * x**2
psiddot = lambda x: 2 * 3 * 20 * np.pi * x

f = lambda x: -20 + a * psiddot(x) * np.cos(psi(x)) - a * psidot(x)**2 * np.sin(psi(x))
u = lambda x: 1 + 12 * x - 10 * x**2 + a * np.sin(psi(x))

# Grid setup
m = 255
h = 1.0 / (m + 1)
X = np.linspace(h, 1 - h, m)

# 1D Laplacian Sparse Matrix A
main_diag = -2.0 * np.ones(m) / h**2
off_diag = np.ones(m - 1) / h**2
A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')

# Right-hand side
F = f(X)
F[0] = F[0] - u(0) / h**2
F[-1] = F[-1] - u(1) / h**2

# Exact solution and theoretical error
Uhat = u(X)
# A\F in MATLAB translates to spla.spsolve in Python
Ehat = spla.spsolve(A, F) - Uhat 

# Jacobi Iteration Matrices
M_diag = A.diagonal()
M_inv = sp.diags(1.0 / M_diag, 0, format='csr')
N = sp.diags(M_diag, 0, format='csr') - A
G = M_inv @ N
b = M_inv @ F

omega = 2.0 / 3.0
U2 = 1 + 2 * X

# --- Plotting Setup ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

def update_plot(ax1, ax2, X, Uhat, U2, Ehat, E2, title_str):
    ax1.clear()
    ax1.plot(X, Uhat, 'b-', label='Exact (Uhat)')
    ax1.plot(X, U2, 'gx', label='Computed (U2)')
    ax1.set_xlabel('x', fontsize=16)
    ax1.set_ylabel('U', fontsize=16)
    ax1.set_title(title_str, fontsize=16)
    
    ax2.clear()
    ax2.plot(X, Ehat, 'b-', label='Exact Error (Ehat)')
    ax2.plot(X, E2, 'gx', label='Computed Error (E2)')
    ax2.set_xlabel('x', fontsize=16)
    ax2.set_ylabel('E', fontsize=16)
    ax2.set_title(title_str, fontsize=16)
    
    plt.draw()
    plt.pause(1.0)

# 1. Pre-smooth the error
print("Starting pre-smoothing...")
for i in range(1, 11):
    U2 = (1 - omega) * U2 + omega * (G @ U2 + b)
    E2 = U2 - Uhat
    update_plot(ax1, ax2, X, Uhat, U2, Ehat, E2, f'Iter={i:4d}')

# Pause before coarse grid correction (equivalent to MATLAB's bare `pause`)
print("Paused. Close the plot window or press Enter in console to continue to coarse grid projection.")
plt.waitforbuttonpress() 



# --- Coarse Grid Correction ---
# Calculate residual
r = F - A @ U2

# Coarsen (Injection)
m_coarse = (m - 1) // 2
h_coarse = 1.0 / (m_coarse + 1)
# MATLAB 2:2:end maps to 1::2 in 0-based Python indexing
r_coarse = r[1::2] 
assert len(r_coarse) == m_coarse

main_diag_c = -2.0 * np.ones(m_coarse) / h_coarse**2
off_diag_c = np.ones(m_coarse - 1) / h_coarse**2
A_coarse = sp.diags([off_diag_c, main_diag_c, off_diag_c], [-1, 0, 1], format='csr')

# Solve the coarse problem directly
e_coarse = spla.spsolve(A_coarse, -r_coarse)

# Project back on the fine grid (Linear Interpolation)
e = np.zeros_like(r)
e[1::2] = e_coarse

# MATLAB 1:2:m maps to 0, 2, 4... in Python
for i in range(0, m, 2):
    e_left = e[i-1] if i > 0 else 0
    e_right = e[i+1] if i < m - 1 else 0
    e[i] = (e_left + e_right) / 2.0

U2 = U2 - e
E2 = U2 - Uhat

update_plot(ax1, ax2, X, Uhat, U2, Ehat, E2, 'After coarse grid projection')

print("Paused. Press a key/click to continue to post-smoothing.")
plt.waitforbuttonpress()

# 2. Post-smooth the error
print("Starting post-smoothing...")
for i in range(1, 11):
    U2 = (1 - omega) * U2 + omega * (G @ U2 + b)
    E2 = U2 - Uhat
    update_plot(ax1, ax2, X, Uhat, U2, Ehat, E2, f'Iter={i:4d}')

plt.ioff()
plt.show()
print("Finished.")