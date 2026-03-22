import numpy as np
import matplotlib.pyplot as plt
from ex3a import Amult

# it plots the max eigenvalue in abs value for p,q in [m/2, m]
def plot_smoothing_factor(m):
    h = 1.0 / (m + 1)
    # we choose many samples for omega
    omega_range = np.linspace(0, 2, 5000)
    
    # grid for p, q indexes
    p = np.arange(1, m + 1)
    q = np.arange(1, m + 1)
    P, Q = np.meshgrid(p, q)
    
    # mask for high frequencies [m/2, m]x[m/2, m]
    high_freq_mask = (P >= m/2) | (Q >= m/2)
    
    # lambda_pq_scaled = 1/2 * [(cos(p*pi*h) - 1) + (cos(q*pi*h) - 1)]
    term_pq = 0.5 * ((np.cos(P * np.pi * h) - 1) + (np.cos(Q * np.pi * h) - 1))
    
    term_high_freq = term_pq[high_freq_mask]
    
    max_gamma = []
    
    for omega in omega_range:
        # gamma = 1 + omega * term_pq
        gamma_high = 1 + omega * term_high_freq
        max_gamma.append(np.max(np.abs(gamma_high)))
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(omega_range, max_gamma, label=r'$\max_{m/2 \leq p,q \leq m} |\gamma_{p,q}|$', color='blue', lw=2)
    
    # value of ω which ”visually minimizes” the max eigenvalue
    omega_opt = omega_range[np.argmin(max_gamma)]
    plt.axvline(omega_opt, color='red', linestyle='--', alpha=0.7, 
                label=fr'Optimal $\omega \approx {omega_opt:.3f}$')
    plt.scatter(omega_opt, min(max_gamma), color='red', zorder=5)

    plt.title(fr'Smoothing Factor analysis for Jacobi (m={m})', fontsize=14)
    plt.xlabel(r'Relaxation parameter $\omega$', fontsize=12)
    plt.ylabel('Max Absolute Eigenvalue (High Frequencies)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.show()
    print(f'for m={m} we obtain as optimal omega: {omega_opt} ')




import numpy as np

def smooth(U, F, m, omega):
    """
    Performs under-relaxed Jacobi smoothing for the 2D Poisson equation.
    
    Args:
        U (np.ndarray): Current approximation of the solution (1D array of size m*m).
        F (np.ndarray): Right-hand side vector (1D array of size m*m).
        m (int): Number of interior grid points in one dimension.
        omega (float, optional): Relaxation parameter. Defaults to 0.8 (optimal for 2D).
        num_iters (int, optional): Number of smoothing sweeps. Defaults to 2.
        
    Returns:
        np.ndarray: The smoothed approximation vector.
    """
    # Calculate grid spacing
    h = 1.0 / (m + 1)
    
    # Inverse of the diagonal matrix D for the 5-point Laplacian.
    D_inv = (h**2) / 4.0  
    
    # Copy U to prevent modifying the original array in-place
    U_k = np.copy(U)
    
    # 1. Compute matrix-vector product A * U_k using the matrix-free function
    AU = Amult(U_k, m)
    
    # 2. Compute the residual R = F - A * U_k
    R = F + AU
    
    # 3. Update the approximation using the relaxed Jacobi formula
    U_k = U_k + omega * D_inv * R
        
    return U_k

# --- TEST SETUP ---
# According to the assignment: m = 2^k - 1
k = 5
m = 2**k - 1  # 31x31 grid
h = 1.0 / (m + 1)

# Create the 2D grid for interior points
x = np.linspace(h, 1-h, m)
y = np.linspace(h, 1-h, m)
X, Y = np.meshgrid(x, y)

# 1. Initial condition: Low-frequency wave + high-frequency noise
low_freq = np.sin(np.pi * X) * np.sin(np.pi * Y)
high_freq = 0.5 * np.sin(15 * np.pi * X) * np.sin(15 * np.pi * Y)
U_initial_2d = low_freq + high_freq

# Flatten to a 1D vector for the function input
U_initial = U_initial_2d.flatten()

# 2. Right-hand side (F) set to 0 to solve the pure Laplace problem
F = np.zeros_like(U_initial)

# 3. Apply the smoother
iterations = 5
U_smoothed = smooth(U_initial, F, m, omega=0.8, num_iters=iterations)

# Reshape back to 2D for plotting
U_smoothed_2d = U_smoothed.reshape((m, m))

# --- PLOT GENERATION ---
fig = plt.figure(figsize=(12, 5))

# Plot 1: Initial Guess
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, U_initial_2d, cmap='viridis', edgecolor='none')
ax1.set_title("Initial Approximation\n(Smooth wave + High-frequency noise)")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('U')

# Plot 2: Smoothed Solution
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, U_smoothed_2d, cmap='viridis', edgecolor='none')
ax2.set_title(f"After {iterations} 'smooth' iterations\n(High-frequency removed)")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('U')
ax2.set_zlim(ax1.get_zlim()) # Keep the same Z-scale for a fair comparison

plt.tight_layout()
# plt.show()

def coarsen(R,m):
    # function body
    return Rc

def interpolate(Rc,m):
    # function body
    return R

"""import numpy as np
import matplotlib.pyplot as plt

# Confrontiamo un m pari e uno dispari vicini
m_vals = [30, 31]
plt.figure(figsize=(10, 6))

# Usiamo un range di omega molto denso per vedere i piccoli spostamenti
omega_range = np.linspace(0.7, 0.9, 1000) 

for m in m_vals:
    h = 1.0 / (m + 1)
    p = np.arange(1, m + 1)
    q = np.arange(1, m + 1)
    P, Q = np.meshgrid(p, q)
    
    # La soglia m/2 cambia comportamento tra pari e dispari
    high_freq_mask = (P >= m/2) | (Q >= m/2)
    
    term_pq = 0.5 * ((np.cos(P * np.pi * h) - 1) + (np.cos(Q * np.pi * h) - 1))
    term_high_freq = term_pq[high_freq_mask]
    
    max_gamma = [np.max(np.abs(1 + omega * term_high_freq)) for omega in omega_range]
    
    idx_min = np.argmin(max_gamma)
    w_opt = omega_range[idx_min]
    
    plt.plot(omega_range, max_gamma, label=fr'm={m} (min at $\omega \approx {w_opt:.4f}$)')
    plt.scatter(w_opt, max_gamma[idx_min], zorder=5)

plt.axvline(0.8, color='black', linestyle='--', alpha=0.3, label='Theoretical 0.8')
plt.title('Comparison of Even vs Odd $m$ (High-Frequency Smoothing)', fontsize=14)
plt.xlabel(r'Relaxation parameter $\omega$', fontsize=12)
plt.ylabel(r'Max $|\gamma_{p,q}|$', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()"""