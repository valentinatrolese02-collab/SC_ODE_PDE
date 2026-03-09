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
    plt.ylim(0, 1.1)
    #plt.show()
    print(f'for m={m} we obtain as optimal omega: {omega_opt} ')

# we experiment different values
ms = [102,97]
for m in ms:
    plot_smoothing_factor(m)

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