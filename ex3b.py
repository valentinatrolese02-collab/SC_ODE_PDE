import numpy as np
import matplotlib.pyplot as plt

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
    plt.ylim(0.40, 1.00)
    plt.xlim(0.40, 1.00)  
    #plt.show()
    print(f'for m={m} we obtain as optimal omega: {omega_opt} ')

# we experiment different values
ms = [30, 40, 50, 100]
for m in ms:
    plot_smoothing_factor(m)

def smooth(U, omega, m, F):
    """
    Matrix-free relaxed Jacobi iteration for the 5-point Laplacian in 2D.
    
    INputs:
    U     : current iterate (of length m^2)
    omega : relaxation parameter
    m     : number of grid points
    F     : right-hand side vector (of length m^2)
    
    Output:
    Unew  : the updated iterate
    """
    h = 1.0 / (m + 1)
    h2 = h**2
    
    # Reshape vectors to 2D grids (it only changes view while remaining matrix-free)
    # U evaluated on m x m internal points => also F is m x m internal points, with implicitey adjusted boundary conditions
    u_grid = U.reshape((m, m))
    f_grid = F.reshape((m, m))
    
    # The term (L+U)u: Sum of the 4 neighbors within the m x m grid
    # Initializing with zeros to accumulate neighbor contributions
    LU_u = np.zeros_like(u_grid)
    
    LU_u[:-1, :] += u_grid[1:, :]  # Down neighbor
    LU_u[1:, :]  += u_grid[:-1, :] # Up neighbor
    LU_u[:, :-1] += u_grid[:, 1:]  # Right neighbor
    LU_u[:, 1:]  += u_grid[:, :-1] # Left neighbor
    
    # Weighted Jacobi Update: (1-w)*U + w * D^-1 * (F - (L+U)U)
    # Since Au = (D - (L+U))u = F => Du = F + (L+U)u
    # Here D = 4/h2, so D^-1 = h2/4
    unew_grid = (1 - omega) * u_grid + (omega / 4.0) * (h2 * f_grid + LU_u)
    
    return unew_grid.flatten()