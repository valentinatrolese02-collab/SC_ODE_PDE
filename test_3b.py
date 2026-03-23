import matplotlib.pyplot as plt
import numpy as np

def coarsen(R, m):
    # Reshape the 1D flat array into a 2D fine grid matrix
    R_mat = R.reshape((m, m))
    
    center = R_mat[1::2, 1::2]
    
    # top:    Starts at 0, stops 2 elements before the end, step 2 (0, 2, 4...)
    # bottom: Starts at 2, goes to the end, step 2 (2, 4, 6...)
    top    = R_mat[0:-2:2, 1::2]
    bottom = R_mat[2::2,    1::2]
    left   = R_mat[1::2,   0:-2:2]
    right  = R_mat[1::2,   2::2]
    
    # Extract the 4 diagonal neighbors
    top_left     = R_mat[0:-2:2, 0:-2:2]
    top_right    = R_mat[0:-2:2, 2::2]
    bottom_left  = R_mat[2::2,    0:-2:2]
    bottom_right = R_mat[2::2,    2::2]
    
    # Apply the full weighting stencil 2D equation in a single vectorized operation
    Rc_mat = (
        4.0 * center + 
        2.0 * (top + bottom + left + right) + 
        1.0 * (top_left + top_right + bottom_left + bottom_right)
    ) / 16.0
    
    Rc = Rc_mat.flatten()
    return Rc

def interpolate(Rc, m):
    mc = int((m - 1) / 2)
    Rc_mat = Rc.reshape((mc, mc))
    
    Rc_pad = np.pad(Rc_mat, pad_width=1, mode='constant', constant_values=0)
    R_mat = np.zeros((m, m))
    
    R_mat[1::2, 1::2] = Rc_mat
    #even rows
    R_mat[0::2, 1::2] = 0.5 * (Rc_pad[:-1, 1:-1] + Rc_pad[1:, 1:-1])
    #even columns
    R_mat[1::2, 0::2] = 0.5 * (Rc_pad[1:-1, :-1] + Rc_pad[1:-1, 1:])
    # even both
    R_mat[0::2, 0::2] = 0.25 * (Rc_pad[:-1, :-1] + Rc_pad[1:, :-1] + 
                                Rc_pad[:-1, 1:] + Rc_pad[1:, 1:])                      
    R=R_mat.flatten()

    return R

def Amult(U, m):
    """
    Matrix-free implementation of the 2D negative Laplacian (-Ah * U).
    Uses array shifting to avoid explicit matrix storage and padding.
    """
    U2d = U.reshape((m, m))
    # this is true if we assume the row-wise ordering

    AU_2d = 4.0 * U2d

    AU_2d[:-1, :] -= U2d[1:, :]  # Down neighbor
    AU_2d[1:, :]  -= U2d[:-1, :] # Up neighbor
    AU_2d[:, :-1] -= U2d[:, 1:]  # Right neighbor
    AU_2d[:, 1:]  -= U2d[:, :-1] # Left neighbor
    
    # Scale by h^2 where h = 1 / (m + 1)
    h2 = (1.0 / (m + 1))**2
    return -(AU_2d / h2).flatten()

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
    h = 1.0 / (m + 1)

    D_inv = (h**2) / 4.0  
    U_k = np.copy(U)
    AU = Amult(U_k, m)
    R = F + AU
    U_k = U_k + omega * D_inv * R
        
    return U_k

def visualize_error_smoothing_and_restriction():
    k = 5
    m = 2**k - 1           # Fine grid (31x31)
    mc = int((m - 1) / 2)  # Coarse grid (15x15)
    
    h = 1.0 / (m + 1)
    hc = 1.0 / (mc + 1)
    
    X, Y = np.meshgrid(np.linspace(h, 1-h, m), np.linspace(h, 1-h, m))
    Xc, Yc = np.meshgrid(np.linspace(hc, 1-hc, mc), np.linspace(hc, 1-hc, mc))
    
    # 1. Create an exact solution (e.g., all zeros for simplicity)
    u_exact = np.zeros(m * m)
    F = np.zeros(m * m) # Right hand side is 0
    
    # 2. Create an initial guess WITH ERROR
    # Error = Low frequency wave + High frequency noise
    low_freq_error = np.sin(np.pi * X) * np.sin(np.pi * Y)
    high_freq_error = 0.5 * np.sin(15 * np.pi * X) * np.sin(15 * np.pi * Y)
    
    initial_error_2d = low_freq_error + high_freq_error
    U_guess = initial_error_2d.flatten() # Since u_exact is 0, U_guess IS the error
    
    # 3. Smooth the error (3 iterations of Jacobi)
    # The high frequencies will be eliminated, leaving only the smooth low frequencies
    U_smoothed = smooth(U_guess, F, m, omega=0.8) # num_iters handled inside or loop it
    for _ in range(2): 
        U_smoothed = smooth(U_smoothed, F, m, omega=0.8)
        
    Error_smoothed_2d = U_smoothed.reshape((m, m))
    
    # 4. Calculate the Residual: r = F - A*U
    # In a real cycle, we restrict the RESIDUAL, not the error directly
    # But for visual purposes, let's restrict the smoothed error itself to see how it looks
    Error_coarse_1d = coarsen(U_smoothed, m)
    Error_coarse_2d = Error_coarse_1d.reshape((mc, mc))
    
    # --- Plotting the Error Lifecycle ---
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: Initial Error (Noisy)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, initial_error_2d, cmap='coolwarm', edgecolor='none')
    ax1.set_title(f'1. Initial Error\nHigh + Low Frequencies ({m}x{m})')
    ax1.set_zlim(-1.5, 1.5)
    
    # Plot 2: Smoothed Error (After Jacobi)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Error_smoothed_2d, cmap='coolwarm', edgecolor='none')
    ax2.set_title(f'2. Smoothed Error\nHigh frequencies removed ({m}x{m})')
    ax2.set_zlim(-1.5, 1.5)
    
    # Plot 3: Coarsened Error (Ready for coarse grid solve)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(Xc, Yc, Error_coarse_2d, cmap='coolwarm', edgecolor='none')
    ax3.set_title(f'3. Restricted Error\nMoved to Coarse Grid ({mc}x{mc})')
    ax3.set_zlim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.show()

# Run the error visualization
visualize_error_smoothing_and_restriction()